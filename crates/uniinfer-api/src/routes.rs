use axum::extract::{DefaultBodyLimit, OriginalUri, Path, State};
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
        .route("/v1/models/{id}/load", post(crate::model_lifecycle::load))
        .route(
            "/v1/models/{id}/unload",
            post(crate::model_lifecycle::unload),
        )
        .route(
            "/v1/models/{id}/manifest",
            get(crate::manifest_api::get).put(crate::manifest_api::put),
        )
        .route("/v1/models/inspect", post(crate::models::inspect_model))
        .route("/v1/models/import", post(crate::models::import_model))
        .route("/v1/models/refresh", post(refresh_models))
        .route("/v1/devices", get(devices))
        .route("/v1/chat/completions", post(chat))
        .route(
            "/v1/config",
            get(crate::config::get).put(crate::config::put),
        )
        .route("/v1/realtime", get(realtime))
        .route("/v1/sessions/{id}/cancel", post(cancel));
    let router = router
        .route("/v1/audio/transcriptions", post(crate::audio::transcribe))
        .route("/v1/audio/speech", post(planned))
        .route("/v1/images/generations", post(crate::images::generate))
        .route("/v1/images/edits", post(planned))
        .route("/v1/videos/generations", post(planned))
        .route("/v1/videos/generations/{id}", get(planned));
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

async fn planned(OriginalUri(uri): OriginalUri) -> Result<Json<Value>, ApiError> {
    Err(ApiError::not_implemented(format!(
        "{} is planned but not implemented",
        uri.path()
    )))
}

async fn metrics(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "active_requests": state.router.active_count(),
        "loaded_sessions": state.runtime.active_sessions(),
        "loaded_models": {
            "text": state.router.slots().len(),
            "audio": state.audio_slots.lock().map(|slots| slots.len()).unwrap_or(0),
            "image": state.image_slots.lock().map(|slots| slots.len()).unwrap_or(0),
        },
        "requests": state.router.metrics(),
    }))
}

fn model_json(slot: &dyn ModelView) -> Value {
    let manifest = slot.manifest();
    json!({
        "id": manifest.id,
        "object": "model",
        "owned_by": "uniinfer",
        "runtime": manifest.runtime,
        "implementation": manifest.implementation,
        "capabilities": manifest.capabilities,
        "format": manifest.format,
        "quantization": manifest.quantization,
        "inputs": manifest.input_modalities(),
        "outputs": manifest.output_modalities(),
        "context_len": manifest.context_len,
        "streaming_capable": manifest.streaming_capable,
        "loaded": true,
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

fn catalog_model_json(state: &AppState, id: &str, entry: &crate::state::CatalogEntry) -> Value {
    if let Some(slot) = state.router.peek_slot(id) {
        return model_json(slot.as_ref());
    }
    if let Some(slot) = state
        .audio_slots
        .lock()
        .ok()
        .and_then(|slots| slots.get(id).cloned())
    {
        return model_json(slot.as_ref());
    }
    if let Some(slot) = state
        .image_slots
        .lock()
        .ok()
        .and_then(|slots| slots.get(id).cloned())
    {
        return model_json(slot.as_ref());
    }
    let manifest = &entry.manifest;
    match state.runtime.probe_manifest(manifest) {
        Ok(route) => json!({
            "id": manifest.id,
            "object": "model",
            "owned_by": "uniinfer",
            "format": manifest.format,
            "quantization": manifest.quantization,
            "inputs": manifest.input_modalities(),
            "outputs": manifest.output_modalities(),
            "context_len": manifest.context_len,
            "streaming_capable": manifest.streaming_capable,
            "loaded": false,
            "engine": route.engine_name,
            "devices": route.placement.devices.iter().map(|device| device.class).collect::<Vec<_>>(),
            "placement": route.placement,
        }),
        Err(error) => json!({
            "id": manifest.id,
            "object": "model",
            "owned_by": "uniinfer",
            "format": manifest.format,
            "quantization": manifest.quantization,
            "inputs": manifest.input_modalities(),
            "outputs": manifest.output_modalities(),
            "context_len": manifest.context_len,
            "streaming_capable": manifest.streaming_capable,
            "loaded": false,
            "engine": Value::Null,
            "devices": [],
            "placement": Value::Null,
            "route_error": {"code": error.code(), "message": error.to_string()},
        }),
    }
}

fn catalog_placement_json(state: &AppState, id: &str, entry: &crate::state::CatalogEntry) -> Value {
    if let Some(slot) = state.router.peek_slot(id) {
        return placement_json(slot.as_ref());
    }
    if let Some(slot) = state
        .audio_slots
        .lock()
        .ok()
        .and_then(|slots| slots.get(id).cloned())
    {
        return placement_json(slot.as_ref());
    }
    match state.runtime.probe_manifest(&entry.manifest) {
        Ok(route) => json!({
            "id": id,
            "loaded": false,
            "engine": route.engine_name,
            "placement": route.placement,
            "fallback_reason": route.placement.fallback_reason,
        }),
        Err(error) => json!({
            "id": id,
            "loaded": false,
            "engine": Value::Null,
            "placement": Value::Null,
            "fallback_reason": Value::Null,
            "route_error": {"code": error.code(), "message": error.to_string()},
        }),
    }
}

async fn refresh_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    state.refresh_catalog()?;
    Ok(Json(json!({"refreshed": true})))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    let data = state
        .catalog
        .read()
        .map(|catalog| {
            catalog
                .iter()
                .map(|(id, entry)| catalog_model_json(&state, id, entry))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
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
                "available_memory_bytes": device.available_memory_bytes,
            })
        })
        .collect::<Vec<_>>();
    let models = state
        .catalog
        .read()
        .map(|catalog| {
            catalog
                .iter()
                .map(|(id, entry)| catalog_placement_json(&state, id, entry))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
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
