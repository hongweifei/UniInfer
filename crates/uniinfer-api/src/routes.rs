use std::sync::atomic::Ordering;

use axum::extract::{Path, State};
use axum::middleware;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::chat::chat;
use crate::error::ApiError;
use crate::state::AppState;

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/sessions/{id}/cancel", post(cancel))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            crate::auth::authorize,
        ))
        .layer(crate::auth::cors())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "active_requests": state.active.load(Ordering::SeqCst),
        "loaded_sessions": state.runtime.active_sessions(),
    }))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let manifest = state.slot.manifest();
    let placement = state.slot.placement();
    Json(json!({
        "object": "list",
        "data": [{
            "id": manifest.id,
            "object": "model",
            "owned_by": "uniinfer",
            "format": manifest.format,
            "quantization": manifest.quantization,
            "modalities": manifest.modalities,
            "context_len": manifest.context_len,
            "streaming_capable": manifest.streaming_capable,
            "devices": manifest.supported_devices,
            "placement": placement,
        }]
    }))
}

async fn cancel(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let token = state
        .cancels
        .lock()
        .ok()
        .and_then(|guard| guard.get(&id).cloned())
        .ok_or_else(|| ApiError::not_found("session_not_found", format!("unknown session {id}")))?;
    token.cancel();
    Ok(Json(json!({"id": id, "cancelled": true})))
}
