use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::Response;
use base64::Engine as _;
use serde::Deserialize;
use uniinfer_backends::{ChatMessage as BackendMessage, Sampling};
use uniinfer_core::{CancelToken, CapabilityRequest, Modality, Mode, Operation};
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
    #[serde(default)]
    model: Option<String>,
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
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
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
    #[serde(default)]
    content: Option<ChatContent>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<serde_json::Value>),
}

#[derive(Debug, Default)]
struct ParsedContent {
    text: String,
    images: Vec<Vec<u8>>,
}

impl ChatContent {
    fn contains_image(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(|part| {
                part.get("type").and_then(serde_json::Value::as_str) == Some("image_url")
            }),
        }
    }

    fn into_parts(self, image_marker: &str) -> Result<ParsedContent, ApiError> {
        match self {
            Self::Text(text) => Ok(ParsedContent {
                text,
                images: Vec::new(),
            }),
            Self::Parts(parts) => {
                let mut content = ParsedContent::default();
                for part in parts {
                    match part.get("type").and_then(serde_json::Value::as_str) {
                        Some("text") => {
                            if let Some(value) =
                                part.get("text").and_then(serde_json::Value::as_str)
                            {
                                content.text.push_str(value);
                            }
                        }
                        Some("image_url") => {
                            content.images.push(decode_image_part(&part)?);
                            content.text.push_str(image_marker);
                        }
                        Some(other) => {
                            return Err(ApiError::bad_request(
                                "unsupported_content",
                                format!("content part {other:?} is not supported"),
                            ));
                        }
                        None => {
                            return Err(ApiError::bad_request(
                                "invalid_content",
                                "content part is missing type",
                            ));
                        }
                    }
                }
                Ok(content)
            }
        }
    }
}

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

fn decode_image_part(part: &serde_json::Value) -> Result<Vec<u8>, ApiError> {
    let image = part.get("image_url").ok_or_else(|| {
        ApiError::bad_request("invalid_content", "image_url part has no image_url")
    })?;
    let url = image
        .as_str()
        .or_else(|| image.get("url").and_then(serde_json::Value::as_str))
        .ok_or_else(|| ApiError::bad_request("invalid_content", "image_url has no string url"))?;
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(ApiError::bad_request(
            "unsupported_image_url",
            "only data: image URLs are supported",
        ));
    };
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err(ApiError::bad_request(
            "invalid_image_url",
            "data URL is missing a comma",
        ));
    };
    if !meta.ends_with(";base64") {
        return Err(ApiError::bad_request(
            "unsupported_image_url",
            "data image must be base64 encoded",
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|_| ApiError::bad_request("invalid_image", "image base64 is invalid"))?;
    if bytes.is_empty() {
        return Err(ApiError::bad_request("invalid_image", "image is empty"));
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(ApiError::bad_request(
            "image_too_large",
            format!("image is {} bytes, limit is {MAX_IMAGE_BYTES}", bytes.len()),
        ));
    }
    Ok(bytes)
}

pub(super) struct Generation {
    pub(super) state: AppState,
    pub(super) slot: Arc<ModelSlot>,
    pub(super) permit: tokio::sync::OwnedSemaphorePermit,
    pub(super) session_id: String,
    pub(super) prompt: String,
    pub(super) images: Vec<Vec<u8>>,
    pub(super) prompt_tokens: usize,
    pub(super) max_tokens: usize,
    pub(super) stop: Vec<String>,
    pub(super) sampling: Sampling,
    pub(super) include_usage: bool,
    pub(super) parse_tool_calls: bool,
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
    let has_image = request
        .messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .any(ChatContent::contains_image);
    let capability = CapabilityRequest {
        operation: Operation::Generate,
        present_inputs: std::iter::once(Modality::Text)
            .chain(has_image.then_some(Modality::Image))
            .collect(),
        requested_outputs: vec![Modality::Text],
        mode: if request.stream {
            Mode::Stream
        } else {
            Mode::Batch
        },
    };
    let model = match request
        .model
        .as_deref()
        .filter(|model| !model.is_empty() && *model != "auto")
    {
        Some(model) => model.to_owned(),
        None => state
            .select_capability(&capability, false, false)
            .ok_or_else(|| {
                ApiError::not_found(
                    "no_capability_match",
                    "no configured model matches this request",
                )
            })?,
    };
    if !state.catalog.contains_key(&model) {
        return Err(ApiError::not_found(
            "model_not_found",
            format!("model {model} is not configured"),
        ));
    }
    let slot = state.text_slot(&model).await.map_err(ApiError::from)?;
    let tool_choice = request.tool_choice.clone();
    let tools = if tool_choice.as_ref().and_then(serde_json::Value::as_str) == Some("none") {
        Vec::new()
    } else {
        request.tools.clone().unwrap_or_default()
    };
    let marker = slot
        .manifest()
        .vision
        .as_ref()
        .and_then(|vision| vision.image_token.clone())
        .unwrap_or_default();
    let mut images = Vec::new();
    let messages = request
        .messages
        .iter()
        .map(|message| {
            let content = message
                .content
                .clone()
                .map(|content| content.into_parts(&marker))
                .transpose()?
                .unwrap_or_default();
            images.extend(content.images);
            Ok(BackendMessage {
                role: message.role.clone(),
                content: content.text,
                tool_calls: message.tool_calls.clone(),
                tool_call_id: message.tool_call_id.clone(),
                name: message.name.clone(),
            })
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    if !slot.manifest().supports(&capability) {
        return Err(ApiError::bad_request(
            "unsupported_capability",
            format!("model {model} does not declare a capability for this request"),
        ));
    }
    if !images.is_empty() {
        if slot.manifest().vision.is_none() {
            return Err(ApiError::bad_request(
                "model_does_not_support_images",
                format!("model {model} has no vision configuration"),
            ));
        }
        if marker.is_empty() {
            return Err(ApiError::bad_request(
                "invalid_vision_config",
                "vision.image_token is not configured for this model",
            ));
        }
        if !slot.supports_image_prompt() && !slot.supports_native_generation() {
            return Err(ApiError::bad_request(
                "unsupported_engine_image",
                format!(
                    "engine {} does not expose image chat through /v1/chat/completions",
                    slot.engine_name()
                ),
            ));
        }
        if images.len() > 1 {
            return Err(ApiError::bad_request(
                "unsupported_content",
                "only one image per request is supported",
            ));
        }
    }
    let prompt = slot
        .render_chat_with_tools(&messages, &tools, tool_choice.as_ref())
        .map_err(ApiError::from)?;
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
    let mut prepared = state
        .router
        .prepare_text(&model, prompt, request.max_tokens)
        .await
        .map_err(ApiError::from)?;
    prepared.images = images;
    let PreparedText {
        slot,
        permit,
        session_id,
        prompt,
        images,
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
        images,
        prompt_tokens,
        max_tokens,
        stop,
        sampling,
        include_usage: request
            .stream_options
            .is_some_and(|options| options.include_usage),
        parse_tool_calls: !tools.is_empty(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(value: serde_json::Value) -> ChatContent {
        serde_json::from_value(value).expect("content")
    }

    #[test]
    fn data_image_is_decoded_and_marker_inserted() {
        let content = parts(serde_json::json!([
            {"type": "text", "text": "look "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": " now"}
        ]))
        .into_parts("<image>")
        .expect("parse");
        assert_eq!(content.text, "look <image> now");
        assert_eq!(content.images, vec![b"hi".to_vec()]);
    }

    #[test]
    fn remote_image_url_is_rejected() {
        let error = parts(serde_json::json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]))
        .into_parts("<image>")
        .expect_err("must reject");
        assert_eq!(error.code, "unsupported_image_url");
    }

    #[test]
    fn invalid_base64_is_rejected() {
        let error = parts(serde_json::json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}}
        ]))
        .into_parts("<image>")
        .expect_err("must reject");
        assert_eq!(error.code, "invalid_image");
    }
}
