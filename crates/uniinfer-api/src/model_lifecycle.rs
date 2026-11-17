//! Explicit model preload/unload endpoints complementing lazy requests.

use axum::Json;
use axum::extract::{Path, State};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::state::AppState;

pub(crate) async fn load(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let entry = state.catalog_entry(&id).ok_or_else(|| {
        ApiError::not_found("model_not_found", format!("model {id} is not configured"))
    })?;
    if entry.audio_only {
        state.audio_slot(&id).await?;
    } else {
        state.text_slot(&id).await?;
    }
    Ok(Json(json!({"id": id, "loaded": true})))
}

pub(crate) async fn unload(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let entry = state.catalog_entry(&id).ok_or_else(|| {
        ApiError::not_found("model_not_found", format!("model {id} is not configured"))
    })?;
    let unloaded = if entry.audio_only {
        state.unload_audio(&id)
    } else if entry.image_only {
        state.unload_image(&id)
    } else {
        state.unload_text(&id)
    };
    Ok(Json(json!({"id": id, "unloaded": unloaded})))
}
