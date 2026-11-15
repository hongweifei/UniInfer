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
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
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
    pub(super) include_usage: bool,
    pub(super) started: std::time::Instant,
    pub(super) deadline: Option<std::time::Instant>,
    pub(super) cancel: CancelToken,
}

pub(crate) async fn chat(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ChatRequest>,
) -> Result<Response, ApiError> {
    let deadline = state
        .request_timeout
        .map(|timeout| std::time::Instant::now() + timeout);
    if !state.catalog.contains_key(&request.model) {
        return Err(ApiError::not_found(
            "model_not_found",
            format!("model {} is not configured", request.model),
        ));
    }
    let slot = state
        .text_slot(&request.model)
        .await
        .map_err(ApiError::from)?;
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
        || request.seed.is_some()
        || request.frequency_penalty.is_some()
        || request.presence_penalty.is_some();
    let sampling = Sampling {
        temperature: request
            .temperature
            .unwrap_or(if sampling_requested { 1.0 } else { 0.0 }),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: request.top_k.unwrap_or(0),
        seed: request.seed,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
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
        started,
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
        include_usage: request
            .stream_options
            .is_some_and(|options| options.include_usage),
        started,
        deadline,
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
