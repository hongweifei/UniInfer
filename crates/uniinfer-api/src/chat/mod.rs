use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use uniinfer_core::CancelToken;
use uniinfer_router::PreparedText;
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
    let prompt = extract_prompt(&request.messages)?;
    let prepared = state
        .router
        .prepare_text(&request.model, prompt, request.max_tokens)
        .await
        .map_err(ApiError::from)?;
    let PreparedText {
        slot,
        permit,
        session_id,
        prompt,
        prompt_tokens,
        max_tokens,
        cancel,
        mut guard,
    } = prepared;

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
