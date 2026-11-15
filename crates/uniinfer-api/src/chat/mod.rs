use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::Response;
use serde::Deserialize;
use uniinfer_backends::{ChatMessage as BackendMessage, Sampling};
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
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(sequence) => vec![sequence],
            Self::Many(sequences) => sequences,
        }
    }
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
    pub(super) stop: Vec<String>,
    pub(super) sampling: Sampling,
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
    let messages = request
        .messages
        .iter()
        .map(|message| BackendMessage {
            role: message.role.clone(),
            content: message.content.clone(),
        })
        .collect::<Vec<_>>();
    let prompt = slot.render_chat(&messages).map_err(ApiError::from)?;
    let stop = request.stop.map(StopInput::into_vec).unwrap_or_default();
    let sampling_requested = request.temperature.is_some()
        || request.top_p.is_some()
        || request.top_k.is_some()
        || request.seed.is_some();
    let sampling = Sampling {
        temperature: request
            .temperature
            .unwrap_or(if sampling_requested { 1.0 } else { 0.0 }),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k.unwrap_or(0),
        seed: request.seed,
    };
    sampling.validate().map_err(ApiError::from)?;
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
        stop,
        sampling,
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

pub(super) fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
