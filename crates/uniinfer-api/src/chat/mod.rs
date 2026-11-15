use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use uniinfer_core::CancelToken;
use uniinfer_service::ModelSlot;

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::state::AppState;

mod complete;
mod stream;

use complete::complete_response;
use stream::stream_response;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    pub(super) max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    role: String,
    content: String,
}

pub(super) struct Generation {
    pub(super) state: AppState,
    pub(super) slot: Arc<ModelSlot>,
    pub(super) permit: tokio::sync::OwnedSemaphorePermit,
    pub(super) session_id: String,
    pub(super) prompt: String,
    pub(super) prompt_tokens: usize,
    pub(super) max_tokens: usize,
    pub(super) cancel: CancelToken,
}

pub(crate) async fn chat(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ChatRequest>,
) -> Result<Response, ApiError> {
    let slot = state.router.slot(&request.model).ok_or_else(|| {
        ApiError::not_found(
            "model_not_found",
            format!("model {} is not loaded", request.model),
        )
    })?;
    let manifest = slot.manifest();

    let prompt = extract_prompt(&request.messages)?;
    let prompt_tokens = slot.count_tokens(&prompt).map_err(ApiError::from)?;
    if prompt_tokens >= manifest.context_len {
        return Err(ApiError::bad_request(
            "context_overflow",
            format!(
                "prompt uses {prompt_tokens} tokens, context_len is {}",
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
        .acquire(state.router.queue_timeout())
        .await
        .map_err(ApiError::from)?;
    let mut guard = state.router.register(&session_id, cancel.clone())?;

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

pub(super) fn extract_prompt(messages: &[ChatMessage]) -> Result<String, ApiError> {
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

pub(super) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_session_id() -> String {
    let counter = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
    format!("sess-{:x}-{counter:x}", unix_seconds())
}
