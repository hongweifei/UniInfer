use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uniinfer_backends::{GenerateOptions, generate_bytes_cancellable};
use uniinfer_core::{CancelToken, EngineError};

use crate::error::ApiError;
use crate::state::{ActiveGuard, AppState};
use crate::utf8::Utf8Decoder;

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/sessions/{id}/cancel", post(cancel))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

enum StreamEvent {
    Delta(String),
    Error(ApiError),
    Done,
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "active_requests": state.active.load(Ordering::SeqCst),
        "loaded_sessions": state.registry.active_sessions(),
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let manifest = &state.slot.manifest;
    let placement = state.slot.session.placement();
    Json(json!({
        "object": "list",
        "data": [{
            "id": manifest.id,
            "object": "model",
            "owned_by": "uniinfer",
            "format": manifest.format,
            "quantization": manifest.quantization,
            "modalities": manifest.modalities,
            "context_len": manifest.context_len,
            "streaming_capable": manifest.streaming_capable,
            "devices": manifest.supported_devices,
            "placement": placement,
        }]
    }))
}

async fn cancel(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let token = state
        .cancels
        .lock()
        .ok()
        .and_then(|guard| guard.get(&id).cloned())
        .ok_or_else(|| ApiError::not_found("session_not_found", format!("unknown session {id}")))?;
    token.cancel();
    Ok(Json(json!({"id": id, "cancelled": true})))
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let manifest = &state.slot.manifest;
    if request.model != manifest.id {
        return Err(ApiError::not_found(
            "model_not_found",
            format!("model {} is not loaded", request.model),
        ));
    }

    let prompt = extract_prompt(&request.messages)?;
    let prompt_tokens = prompt.len();
    if prompt_tokens >= manifest.context_len {
        return Err(ApiError::bad_request(
            "context_overflow",
            format!(
                "prompt uses {prompt_tokens} bytes, context_len is {}",
                manifest.context_len
            ),
        ));
    }

    let requested = request.max_tokens.unwrap_or(16);
    let max_tokens = requested.min(manifest.context_len - prompt_tokens);
    if max_tokens == 0 {
        return Err(ApiError::bad_request(
            "invalid_max_tokens",
            "max_tokens resolves to zero",
        ));
    }

    let text = manifest.text.as_ref().ok_or_else(|| {
        ApiError::bad_request("unsupported_model", "manifest has no [text] section")
    })?;
    let vocab = text.vocab_size;
    let options = GenerateOptions {
        max_tokens,
        eos_token: text.eos_token,
    };
    let context_len = manifest.context_len;

    let session_id = next_session_id();
    let cancel = CancelToken::new();
    let permit = match timeout(
        state.queue_timeout,
        state.slot.permit.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => {
            state.unregister(&session_id);
            return Err(ApiError::internal("model semaphore closed"));
        }
        Err(_) => {
            return Err(ApiError::busy("model queue timeout"));
        }
    };
    state.register(&session_id, cancel.clone())?;

    if request.stream {
        stream_response(
            state,
            permit,
            session_id,
            prompt,
            vocab,
            context_len,
            options,
            cancel,
        )
        .await
    } else {
        complete_response(
            state,
            permit,
            session_id,
            prompt,
            prompt_tokens,
            vocab,
            context_len,
            options,
            cancel,
        )
        .await
    }
}

fn extract_prompt(messages: &[ChatMessage]) -> Result<String, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_request",
            "messages is empty",
        ));
    }
    let message = messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .unwrap_or_else(|| messages.last().expect("non-empty"));
    if message.content.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_request",
            "message content is empty",
        ));
    }
    Ok(message.content.clone())
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    state: AppState,
    permit: tokio::sync::OwnedSemaphorePermit,
    session_id: String,
    prompt: String,
    vocab: usize,
    context_len: usize,
    options: GenerateOptions,
    cancel: CancelToken,
) -> Result<Response, ApiError> {
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);
    let model = state.slot.manifest.id.clone();
    let created = unix_seconds();
    let task_state = state.clone();
    let task_session = session_id.clone();

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = ActiveGuard::new(task_state.active.clone());
        let mut decoder = Utf8Decoder::new();
        let result = generate_bytes_cancellable(
            task_state.slot.session.session(),
            &prompt,
            vocab,
            context_len,
            &options,
            &cancel,
            &mut |byte| match decoder.push(byte) {
                Some(text) => tx.blocking_send(StreamEvent::Delta(text)).is_ok(),
                None => true,
            },
        );
        let tail = decoder.finish();
        if !tail.is_empty() {
            let _ = tx.blocking_send(StreamEvent::Delta(tail));
        }
        match result {
            Ok(_) | Err(EngineError::Cancelled) => {}
            Err(error) => {
                let _ = tx.blocking_send(StreamEvent::Error(error.into()));
            }
        }
        let _ = tx.blocking_send(StreamEvent::Done);
        task_state.unregister(&task_session);
    });

    let stream_model = model.clone();
    let stream_id = session_id.clone();
    let stream = ReceiverStream::new(rx).map(move |event| -> Result<Event, Infallible> {
        Ok(match event {
            StreamEvent::Delta(text) => {
                Event::default().data(chunk_json(&stream_id, &stream_model, created, &text))
            }
            StreamEvent::Error(error) => Event::default()
                .event("error")
                .data(json!({"code": error.code, "message": error.message}).to_string()),
            StreamEvent::Done => Event::default().data("[DONE]"),
        })
    });

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response();
    response.headers_mut().insert(
        "x-uniinfer-session-id",
        HeaderValue::from_str(&session_id).map_err(|_| ApiError::internal("invalid session id"))?,
    );
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
async fn complete_response(
    state: AppState,
    permit: tokio::sync::OwnedSemaphorePermit,
    session_id: String,
    prompt: String,
    prompt_tokens: usize,
    vocab: usize,
    context_len: usize,
    options: GenerateOptions,
    cancel: CancelToken,
) -> Result<Response, ApiError> {
    let task_state = state.clone();
    let task_session = session_id.clone();
    let (result, bytes) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = ActiveGuard::new(task_state.active.clone());
        let mut bytes = Vec::new();
        let result = generate_bytes_cancellable(
            task_state.slot.session.session(),
            &prompt,
            vocab,
            context_len,
            &options,
            &cancel,
            &mut |byte| {
                bytes.push(byte);
                true
            },
        );
        task_state.unregister(&task_session);
        (result, bytes)
    })
    .await
    .map_err(|_| ApiError::internal("generation task panicked"))?;

    if let Err(error) = result {
        return Err(error.into());
    }

    let content = String::from_utf8_lossy(&bytes).into_owned();
    let created = unix_seconds();
    Ok(Json(json!({
        "id": format!("chatcmpl-{session_id}"),
        "object": "chat.completion",
        "created": created,
        "model": state.slot.manifest.id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": bytes.len(),
            "total_tokens": prompt_tokens + bytes.len()
        }
    }))
    .into_response())
}

fn chunk_json(id: &str, model: &str, created: u64, text: &str) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": null
        }]
    })
    .to_string()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

fn next_session_id() -> String {
    let counter = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
    format!("sess-{:x}-{counter:x}", unix_seconds())
}
