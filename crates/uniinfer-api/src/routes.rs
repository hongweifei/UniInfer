use axum::extract::{DefaultBodyLimit, Path, State};
use axum::middleware;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use uniinfer_service::ModelView;

use crate::chat::chat;
use crate::error::ApiError;
use crate::realtime::realtime;
use crate::state::AppState;

/// Build the HTTP router.
pub fn router(state: AppState) -> Router {
    let max_body_bytes = state.max_body_bytes;
    let router = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/models/inspect", post(crate::models::inspect_model))
        .route("/v1/models/import", post(crate::models::import_model))
        .route("/v1/devices", get(devices))
        .route("/v1/chat/completions", post(chat))
        .route(
            "/v1/config",
            get(crate::config::get).put(crate::config::put),
        )
        .route("/v1/realtime", get(realtime))
        .route("/v1/sessions/{id}/cancel", post(cancel));
    let router = router.route("/v1/audio/transcriptions", post(crate::audio::transcribe));
    router
        .layer(DefaultBodyLimit::max(max_body_bytes))
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
        "active_requests": state.router.active_count(),
        "loaded_sessions": state.runtime.active_sessions(),
    }))
}

fn model_json(slot: &dyn ModelView) -> Value {
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
        "engine": slot.engine_name(),
        "devices": slot.placement().devices.iter().map(|device| device.class).collect::<Vec<_>>(),
        "placement": slot.placement(),
    })
}

fn placement_json(slot: &dyn ModelView) -> Value {
    json!({
        "id": slot.manifest().id,
        "engine": slot.engine_name(),
        "placement": slot.placement(),
        "fallback_reason": slot.placement().fallback_reason,
    })
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let data = state
        .router
        .slots()
        .values()
        .map(|slot| model_json(slot.as_ref()))
        .collect::<Vec<_>>();
    let mut data = data;
    data.extend(
        state
            .audio_slots
            .values()
            .map(|slot| model_json(slot.as_ref())),
    );
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
        .router
        .slots()
        .values()
        .map(|slot| placement_json(slot.as_ref()))
        .collect::<Vec<_>>();
    let mut models = models;
    models.extend(
        state
            .audio_slots
            .values()
            .map(|slot| placement_json(slot.as_ref())),
    );
    Json(json!({"devices": devices, "models": models}))
}

async fn cancel(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    if !state.router.cancel(&id) {
        return Err(ApiError::not_found(
            "session_not_found",
            format!("unknown session {id}"),
        ));
    }
    Ok(Json(json!({"id": id, "cancelled": true})))
}
