//! Durable configuration read/write for the future GUI.

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};
use uniinfer_config::UniInferConfig;

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::state::AppState;

const REDACTED_TOKEN: &str = "<redacted>";

pub(crate) async fn get(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let handle = state
        .config
        .as_ref()
        .ok_or_else(|| ApiError::not_found("config_not_available", "server has no config file"))?;
    let config = handle.snapshot().map_err(ApiError::from)?;
    let token_configured = config.server.token.is_some();
    let mut value = serde_json::to_value(&config)
        .map_err(|error| ApiError::internal(format!("serialize config: {error}")))?;
    if let Some(server) = value.get_mut("server").and_then(Value::as_object_mut) {
        if token_configured {
            server.insert("token".into(), Value::String(REDACTED_TOKEN.into()));
        }
    }
    Ok(Json(json!({
        "path": handle.path(),
        "token_configured": token_configured,
        "config": value,
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
    let mut config = config;
    let previous = handle.snapshot().map_err(ApiError::from)?;
    if config.server.token.as_deref() == Some(REDACTED_TOKEN) {
        config.server.token = previous.server.token.clone();
    }
    let restart_required = previous.server.addr != config.server.addr
        || previous.server.token != config.server.token
        || previous.server.max_body_bytes != config.server.max_body_bytes
        || previous.server.isolate_models != config.server.isolate_models;
    handle.replace(config.clone()).map_err(ApiError::from)?;
    state.apply_server_config(&config.server);
    Ok(Json(json!({
        "saved": true,
        "restart_required": restart_required,
        "path": handle.path(),
    })))
}
