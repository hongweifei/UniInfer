//! OpenAI-compatible image generation endpoint.

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use uniinfer_backends::{ImageOutput, ModelOutput};
use uniinfer_core::{CapabilityRequest, Modality, Mode, Operation};
use uniinfer_service::{ImageGenerationSpec, ModelExecutor, ModelRequest};

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub(crate) struct ImageRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
}

pub(crate) async fn generate(
    State(state): State<AppState>,
    Json(request): Json<ImageRequest>,
) -> Result<Response, ApiError> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::bad_request(
            "invalid_prompt",
            "prompt is required",
        ));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request(
            "unsupported_n",
            "only one image per request is supported",
        ));
    }
    if let Some(format) = request.response_format.as_deref()
        && format != "b64_json"
    {
        return Err(ApiError::bad_request(
            "unsupported_response_format",
            "only b64_json is supported",
        ));
    }
    let capability = CapabilityRequest {
        operation: Operation::Generate,
        present_inputs: vec![Modality::Text],
        requested_outputs: vec![Modality::Image],
        mode: Mode::Batch,
    };
    let model = match request
        .model
        .as_deref()
        .filter(|model| !model.is_empty() && *model != "auto")
    {
        Some(model) => model.to_owned(),
        None => state
            .select_capability(&capability, false, true)
            .ok_or_else(|| {
                ApiError::not_found(
                    "no_capability_match",
                    "no configured model matches text-to-image generation",
                )
            })?,
    };
    let entry = state.catalog_entry(&model).ok_or_else(|| {
        ApiError::not_found(
            "model_not_found",
            format!("model {model} is not configured"),
        )
    })?;
    if !entry.manifest.supports(&capability) {
        return Err(ApiError::bad_request(
            "unsupported_capability",
            format!("model {model} does not declare text-to-image generation"),
        ));
    }
    let (width, height) = request
        .size
        .as_deref()
        .map(parse_size)
        .transpose()?
        .unwrap_or((0, 0));
    let slot = state.image_slot(&model).await.map_err(ApiError::from)?;
    let defaults = slot.image_config();
    let spec = ImageGenerationSpec {
        prompt: request.prompt,
        negative_prompt: request.negative_prompt,
        width: if width == 0 { defaults.width } else { width },
        height: if height == 0 { defaults.height } else { height },
        steps: request.steps.unwrap_or(1),
        guidance: request.guidance.unwrap_or(0.0),
        seed: request.seed,
    };
    let permit = slot
        .acquire(state.router.queue_timeout())
        .await
        .map_err(ApiError::from)?;
    let output = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        slot.execute(
            ModelRequest::Image { spec },
            &uniinfer_core::CancelToken::new(),
            &mut |_| {},
        )
    })
    .await
    .map_err(|_| ApiError::internal("image generation task panicked"))?
    .map_err(ApiError::from)?;
    let image = match output {
        ModelOutput::Image(image) => image,
        other => {
            return Err(ApiError::internal(format!(
                "image generation returned {:?}",
                other.modalities()
            )));
        }
    };
    Ok(Json(image_json(image)).into_response())
}

fn image_json(image: ImageOutput) -> serde_json::Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(&image.bytes);
    serde_json::json!({
        "created": crate::chat::unix_seconds(),
        "data": [{
            "b64_json": b64,
            "mime_type": image.mime,
            "width": image.width,
            "height": image.height
        }]
    })
}

fn parse_size(value: &str) -> Result<(u32, u32), ApiError> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| ApiError::bad_request("invalid_size", "size must be WxH"))?;
    let width = width
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("invalid_size", "invalid width"))?;
    let height = height
        .parse::<u32>()
        .map_err(|_| ApiError::bad_request("invalid_size", "invalid height"))?;
    if width == 0 || height == 0 || width > 4096 || height > 4096 {
        return Err(ApiError::bad_request(
            "invalid_size",
            "size must be within 1..=4096",
        ));
    }
    Ok((width, height))
}
