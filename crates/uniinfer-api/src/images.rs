//! OpenAI-compatible image generation endpoint.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use tokio::sync::{OwnedSemaphorePermit, mpsc};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uniinfer_backends::{ImageOutput, ModelEvent, ModelOutput};
use uniinfer_core::{CancelToken, CapabilityRequest, Modality, Mode, Operation};
use uniinfer_service::{ImageGenerationSpec, ImageSlot, ModelExecutor, ModelRequest};

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
    /// Emit SSE progress and per-step preview images.
    #[serde(default)]
    stream: bool,
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
        steps: request.steps.unwrap_or_else(|| {
            defaults
                .diffusion
                .as_ref()
                .map_or(1, |diffusion| diffusion.steps)
        }),
        guidance: request.guidance.unwrap_or_else(|| {
            defaults
                .diffusion
                .as_ref()
                .map_or(0.0, |diffusion| diffusion.guidance)
        }),
        seed: request.seed,
        preview: request.stream,
    };
    let permit = slot
        .acquire(state.router.queue_timeout())
        .await
        .map_err(ApiError::from)?;
    if request.stream {
        return stream_response(slot, spec, permit);
    }
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
    let image = output
        .into_outputs()
        .into_iter()
        .find_map(|output| match output {
            ModelOutput::Image(image) => Some(image),
            _ => None,
        })
        .ok_or_else(|| ApiError::internal("image generation returned no image"))?;
    Ok(Json(image_json(image)).into_response())
}

enum ImageStreamEvent {
    Progress {
        step: usize,
        total: usize,
    },
    Preview {
        step: usize,
        total: usize,
        image: ImageOutput,
    },
    Done(ImageOutput),
    Error(ApiError),
}

fn stream_response(
    slot: Arc<ImageSlot>,
    spec: ImageGenerationSpec,
    permit: OwnedSemaphorePermit,
) -> Result<Response, ApiError> {
    let (tx, rx) = mpsc::channel::<ImageStreamEvent>(8);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let cancel = CancelToken::new();
        let send_error = |error: ApiError| {
            let _ = tx.blocking_send(ImageStreamEvent::Error(error));
        };
        let result = slot.execute(
            ModelRequest::Image { spec },
            &cancel,
            &mut |event| match event {
                ModelEvent::Progress { step, total } => {
                    if tx
                        .blocking_send(ImageStreamEvent::Progress { step, total })
                        .is_err()
                    {
                        cancel.cancel();
                    }
                }
                ModelEvent::Preview { step, total, image } => {
                    if tx
                        .blocking_send(ImageStreamEvent::Preview { step, total, image })
                        .is_err()
                    {
                        cancel.cancel();
                    }
                }
                _ => {}
            },
        );
        match result {
            Ok(ModelOutput::Image(image)) => {
                let _ = tx.blocking_send(ImageStreamEvent::Done(image));
            }
            Ok(_) => send_error(ApiError::internal("image generation returned no image")),
            Err(error) => send_error(error.into()),
        }
    });

    let stream = ReceiverStream::new(rx).map(|event| {
        Ok::<_, Infallible>(match event {
            ImageStreamEvent::Progress { step, total } => Event::default().event("progress").data(
                serde_json::json!({"type": "progress", "step": step, "total": total}).to_string(),
            ),
            ImageStreamEvent::Preview { step, total, image } => {
                Event::default().event("preview").data(
                    serde_json::json!({
                        "type": "preview",
                        "step": step,
                        "total": total,
                        "image": image_payload(&image),
                    })
                    .to_string(),
                )
            }
            ImageStreamEvent::Done(image) => Event::default().event("done").data(
                serde_json::json!({"type": "done", "data": [image_payload(&image)]}).to_string(),
            ),
            ImageStreamEvent::Error(error) => Event::default().event("error").data(
                serde_json::json!({
                    "type": "error",
                    "code": error.code,
                    "message": error.message,
                })
                .to_string(),
            ),
        })
    });
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response();
    response
        .headers_mut()
        .insert("cache-control", "no-cache".parse().expect("header"));
    Ok(response)
}

fn image_payload(image: &ImageOutput) -> serde_json::Value {
    let b64 = base64::engine::general_purpose::STANDARD.encode(&image.bytes);
    serde_json::json!({
        "b64_json": b64,
        "mime_type": image.mime,
        "width": image.width,
        "height": image.height
    })
}

fn image_json(image: ImageOutput) -> serde_json::Value {
    serde_json::json!({
        "created": crate::chat::unix_seconds(),
        "data": [image_payload(&image)]
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
