use std::sync::atomic::Ordering;

use axum::extract::{Path, State};
use axum::middleware;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::chat::chat;
use crate::error::ApiError;
use crate::realtime::realtime;
use crate::state::AppState;

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/devices", get(devices))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/realtime", get(realtime))
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
    let data = state
        .slots
        .values()
        .map(|slot| {
            let manifest = slot.manifest();
            json!({
                "id": manifest.id,
                "object": "model",
                "owned_by": "uniinfer",
                "format": manifest.format,
                "quantization": manifest.quantization,
                "modalities": manifest.modalities,
                "context_len": manifest.context_len,
                "streaming_capable": manifest.streaming_capable,
                "devices": slot.placement().devices.iter().map(|device| device.class).collect::<Vec<_>>(),
                "placement": slot.placement(),
            })
        })
        .collect::<Vec<_>>();
    Json(json!({"object": "list", "data": data}))
}

async fn devices(State(state): State<AppState>) -> Json<Value> {
    let devices = state
        .runtime
        .devices()
        .iter()
        .map(|device| {
            json!({
                "id": device.id,
                "class": device.class,
                "backend": device.backend,
                "memory_bytes": device.memory_bytes,
            })
        })
        .collect::<Vec<_>>();
    let models = state
        .slots
        .values()
        .map(|slot| {
            json!({
                "id": slot.manifest().id,
                "engine": slot.engine_name(),
                "placement": slot.placement(),
                "fallback_reason": slot.placement().fallback_reason,
            })
        })
        .collect::<Vec<_>>();
    Json(json!({"devices": devices, "models": models}))
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
