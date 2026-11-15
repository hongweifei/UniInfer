//! Durable configuration read/write for the future GUI.

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};
use uniinfer_config::UniInferConfig;

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::state::AppState;

pub(crate) async fn get(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let handle = state
        .config
        .as_ref()
        .ok_or_else(|| ApiError::not_found("config_not_available", "server has no config file"))?;
    let config = handle.snapshot().map_err(ApiError::from)?;
    Ok(Json(json!({
        "path": handle.path(),
        "config": config,
    })))
}

pub(crate) async fn put(
    State(state): State<AppState>,
    ApiJson(config): ApiJson<UniInferConfig>,
) -> Result<Json<Value>, ApiError> {
    let handle = state
        .config
        .as_ref()
        .ok_or_else(|| ApiError::not_found("config_not_available", "server has no config file"))?;
    handle.replace(config).map_err(ApiError::from)?;
    Ok(Json(json!({
        "saved": true,
        "restart_required": true,
        "path": handle.path(),
    })))
}
