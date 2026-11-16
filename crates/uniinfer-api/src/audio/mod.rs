//! OpenAI-compatible audio transcription endpoint.

use axum::extract::{Multipart, State};
use axum::response::{IntoResponse, Response};
use uniinfer_backends::ModelOutput;
use uniinfer_backends::audio::decode_wav_mono;
use uniinfer_core::{CapabilityRequest, EngineError, Modality, Mode, Operation};
use uniinfer_service::{ModelExecutor, ModelRequest};

use crate::error::ApiError;
use crate::state::AppState;

mod response;
use response::ResponseFormat;

pub(crate) async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut model = None;
    let mut audio = None;
    let mut format = ResponseFormat::Json;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::bad_request("invalid_multipart", error.to_string()))?
    {
        match field.name().unwrap_or_default() {
            "model" => {
                model =
                    Some(field.text().await.map_err(|error| {
                        ApiError::bad_request("invalid_model", error.to_string())
                    })?);
            }
            "response_format" => {
                let value = field.text().await.map_err(|error| {
                    ApiError::bad_request("invalid_response_format", error.to_string())
                })?;
                format = ResponseFormat::parse(&value).ok_or_else(|| {
                    ApiError::bad_request(
                        "invalid_response_format",
                        format!("unsupported response_format {value:?}"),
                    )
                })?;
            }
            "file" | "audio" => {
                audio =
                    Some(field.bytes().await.map_err(|error| {
                        ApiError::bad_request("invalid_audio", error.to_string())
                    })?);
            }
            _ => {}
        }
    }

    let audio = audio.ok_or_else(|| ApiError::bad_request("missing_audio", "file is required"))?;
    let capability = CapabilityRequest {
        operation: Operation::Transcribe,
        present_inputs: vec![Modality::Audio],
        requested_outputs: vec![Modality::Text],
        mode: Mode::Batch,
    };
    let model = match model.filter(|model| !model.is_empty() && model != "auto") {
        Some(model) => model,
        None => state
            .select_capability(&capability, true, false)
            .ok_or_else(|| {
                ApiError::not_found(
                    "no_capability_match",
                    "no configured model matches audio transcription",
                )
            })?,
    };
    let entry = state.catalog_entry(&model).ok_or_else(|| {
        ApiError::not_found(
            "model_not_found",
            format!("model {model} is not configured"),
        )
    })?;
    if !entry.audio_only {
        return Err(ApiError::bad_request(
            "model_not_audio",
            format!("model {model} is not an audio-only model"),
        ));
    }
    if !entry.manifest.supports(&capability) {
        return Err(ApiError::bad_request(
            "unsupported_capability",
            format!("model {model} does not declare audio transcription"),
        ));
    }
    let slot = state.audio_slot(&model).await.map_err(ApiError::from)?;
    let samples = decode_wav_mono(&audio, slot.audio_config().sample_rate)?;
    let audio_seconds = samples.len() as f64 / f64::from(slot.audio_config().sample_rate);
    let permit = slot.acquire(state.router.queue_timeout()).await?;
    let started = std::time::Instant::now();
    let process_isolated = slot.engine_name() == "process";
    let deadline = state.request_timeout.map(|timeout| started + timeout);
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        slot.execute(
            ModelRequest::Audio { samples, deadline },
            &uniinfer_core::CancelToken::new(),
            &mut |_| {},
        )
    })
    .await
    .map_err(|_| ApiError::internal("transcription task panicked"))?;
    state
        .router
        .record_asr(&model, audio_seconds, started.elapsed());
    let text = match result {
        Ok(output) => output
            .into_outputs()
            .into_iter()
            .find_map(|output| match output {
                ModelOutput::Text(text) => Some(text),
                _ => None,
            })
            .ok_or_else(|| ApiError::internal("transcription returned no text"))?,
        Err(error) => {
            if matches!(error, EngineError::Timeout) {
                state.router.record_timeout(&model);
            }
            if process_isolated
                && matches!(error, EngineError::Timeout | EngineError::ProcessExited)
            {
                state.unload_audio(&model);
            }
            return Err(error.into());
        }
    };
    Ok(response::render(format, &text, audio_seconds).into_response())
}
