//! OpenAI-compatible audio transcription endpoint.

use axum::Json;
use axum::extract::{Multipart, State};
use serde_json::{Value, json};
use uniinfer_backends::audio::decode_wav_mono;

use crate::error::ApiError;
use crate::state::AppState;

pub(crate) async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, ApiError> {
    let mut model = None;
    let mut audio = None;
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
            "file" | "audio" => {
                audio =
                    Some(field.bytes().await.map_err(|error| {
                        ApiError::bad_request("invalid_audio", error.to_string())
                    })?);
            }
            _ => {}
        }
    }

    let model = model.ok_or_else(|| ApiError::bad_request("missing_model", "model is required"))?;
    let audio = audio.ok_or_else(|| ApiError::bad_request("missing_audio", "file is required"))?;
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
    let slot = state.audio_slot(&model).await.map_err(ApiError::from)?;
    let samples = decode_wav_mono(&audio, slot.audio_config().sample_rate)?;
    let audio_seconds = samples.len() as f64 / f64::from(slot.audio_config().sample_rate);
    let permit = slot.acquire(state.router.queue_timeout()).await?;
    let started = std::time::Instant::now();
    let text = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        slot.transcribe(samples)
    })
    .await
    .map_err(|_| ApiError::internal("transcription task panicked"))??;
    state
        .router
        .record_asr(&model, audio_seconds, started.elapsed());
    Ok(Json(json!({"text": text})))
}
