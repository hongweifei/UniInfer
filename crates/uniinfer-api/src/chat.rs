use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio::sync::mpsc;
use uniinfer_core::{CancelToken, EngineError};
use uniinfer_service::ModelSlot;

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::sse::{self, StreamEvent};
use crate::state::{ActiveGuard, AppState, CancelGuard};
use crate::utf8::Utf8Decoder;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    role: String,
    content: String,
}

struct Generation {
    state: AppState,
    slot: Arc<ModelSlot>,
    permit: tokio::sync::OwnedSemaphorePermit,
    session_id: String,
    prompt: String,
    prompt_tokens: usize,
    max_tokens: usize,
    cancel: CancelToken,
}

pub(crate) async fn chat(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ChatRequest>,
) -> Result<Response, ApiError> {
    let slot = state.slots.get(&request.model).cloned().ok_or_else(|| {
        ApiError::not_found(
            "model_not_found",
            format!("model {} is not loaded", request.model),
        )
    })?;
    let manifest = slot.manifest();

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

    let session_id = next_session_id();
    let cancel = CancelToken::new();
    let permit = slot
        .acquire(state.queue_timeout)
        .await
        .map_err(ApiError::from)?;
    let mut guard = CancelGuard::new(&state, session_id.clone(), cancel.clone())?;

    let generation = Generation {
        state,
        slot,
        permit,
        session_id,
        prompt,
        prompt_tokens,
        max_tokens,
        cancel,
    };

    if request.stream {
        let mut generation = generation;
        generation.cancel = guard.token();
        let response = stream_response(generation).await;
        guard.disarm();
        response
    } else {
        complete_response(generation).await
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

async fn stream_response(generation: Generation) -> Result<Response, ApiError> {
    let Generation {
        state,
        slot,
        permit,
        session_id,
        prompt,
        max_tokens,
        cancel,
        ..
    } = generation;
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);
    let model = slot.manifest().id.clone();
    let created = unix_seconds();
    let task_state = state.clone();
    let task_session = session_id.clone();
    let task_slot = Arc::clone(&slot);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = ActiveGuard::new(task_state.active.clone());
        let mut decoder = Utf8Decoder::new();
        let result = task_slot.generate(&prompt, max_tokens, &cancel, &mut |chunk| {
            for &byte in chunk {
                if let Some(text) = decoder.push(byte)
                    && tx.blocking_send(StreamEvent::Delta(text)).is_err()
                {
                    return false;
                }
            }
            true
        });
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
        let _ = tx.blocking_send(StreamEvent::Finish);
        let _ = tx.blocking_send(StreamEvent::Done);
        task_state.unregister(&task_session);
    });

    let stream = sse::stream(rx, session_id.clone(), model, created);
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response();
    response.headers_mut().insert(
        "x-uniinfer-session-id",
        HeaderValue::from_str(&session_id).map_err(|_| ApiError::internal("invalid session id"))?,
    );
    Ok(response)
}

async fn complete_response(generation: Generation) -> Result<Response, ApiError> {
    let Generation {
        state,
        slot,
        permit,
        session_id,
        prompt,
        prompt_tokens,
        max_tokens,
        cancel,
    } = generation;
    let task_state = state.clone();
    let task_session = session_id.clone();
    let model = slot.manifest().id.clone();
    let (result, bytes) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = ActiveGuard::new(task_state.active.clone());
        let mut bytes = Vec::new();
        let result = slot.generate(&prompt, max_tokens, &cancel, &mut |chunk| {
            bytes.extend_from_slice(chunk);
            true
        });
        task_state.unregister(&task_session);
        (result, bytes)
    })
    .await
    .map_err(|_| ApiError::internal("generation task panicked"))?;

    if let Err(error) = result {
        return Err(error.into());
    }

    let content = String::from_utf8_lossy(&bytes).into_owned();
    Ok(Json(serde_json::json!({
        "id": format!("chatcmpl-{session_id}"),
        "object": "chat.completion",
        "created": unix_seconds(),
        "model": model,
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
