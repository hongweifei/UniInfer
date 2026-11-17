//! Unified list and selection of built-in and sidecar inference backends.

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;
use serde_json::{Value, json};
use uniinfer_backend_store::{
    BUILTIN_BACKEND_ID, BackendEngine, BackendInfo, BackendKind, BackendStore, InstalledBackend,
};
use uniinfer_registry::EngineDescriptor;

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::state::AppState;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ActivateRequest {
    #[serde(default)]
    version: Option<String>,
}

pub(crate) async fn list(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let installed = state
        .backends
        .installed()
        .map_err(|error| ApiError::internal(format!("read backend store: {error}")))?;
    let selected = selected_backend(&state)?;
    let running = running_backend(&state);
    let mut backends = state
        .runtime
        .engine_descriptors()
        .into_iter()
        .map(builtin_info)
        .collect::<Vec<_>>();
    backends.extend(installed.iter().map(InstalledBackend::info));
    let data = backends
        .into_iter()
        .map(|backend| {
            let active = if backend.kind == BackendKind::Builtin {
                selected
                    .as_ref()
                    .is_some_and(|(id, _)| id == BUILTIN_BACKEND_ID)
            } else {
                selected
                    .as_ref()
                    .is_some_and(|(id, version)| id == &backend.id && version == &backend.version)
            };
            json!({
                "id": backend.id,
                "name": backend.name,
                "version": backend.version,
                "description": backend.description,
                "platform": backend.platform,
                "devices": backend.devices,
                "kind": backend.kind,
                "size_bytes": backend.size_bytes,
                "engines": backend.engines,
                "active": active,
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "active": selected.map(|(id, version)| backend_ref(&id, &version)),
        "running": backend_ref(&running.0, &running.1),
        "directory": state.backends.root(),
        "backends": data,
    })))
}

pub(crate) async fn activate(
    Path(id): Path<String>,
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ActivateRequest>,
) -> Result<Json<Value>, ApiError> {
    if id == BUILTIN_BACKEND_ID {
        state
            .backends
            .clear_active()
            .map_err(|error| ApiError::internal(format!("clear active backend: {error}")))?;
        return Ok(Json(json!({
            "active": {"id": BUILTIN_BACKEND_ID, "version": env!("CARGO_PKG_VERSION"), "kind": "builtin"},
            "restart_required": true,
        })));
    }
    let version = match request.version {
        Some(version) => version,
        None => latest_version(&state.backends, &id)?,
    };
    state
        .backends
        .set_active(&id, &version)
        .map_err(|error| ApiError::bad_request("backend_activate_failed", error.to_string()))?;
    Ok(Json(json!({
        "active": {"id": id, "version": version, "kind": "sidecar"},
        "restart_required": true,
    })))
}

pub(crate) async fn remove(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    if id == BUILTIN_BACKEND_ID {
        return Err(ApiError::bad_request(
            "builtin_backend",
            "the built-in backend cannot be removed",
        ));
    }
    let removed = state
        .backends
        .remove(&id)
        .map_err(|error| ApiError::bad_request("backend_remove_failed", error.to_string()))?;
    Ok(Json(json!({
        "id": id,
        "removed": removed,
        "restart_required": true,
    })))
}

/// Kept as an alias for clients that predate the unified `builtin` entry.
pub(crate) async fn deactivate(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    state
        .backends
        .clear_active()
        .map_err(|error| ApiError::internal(format!("clear active backend: {error}")))?;
    Ok(Json(json!({
        "active": {"id": BUILTIN_BACKEND_ID, "version": env!("CARGO_PKG_VERSION"), "kind": "builtin"},
        "restart_required": true,
    })))
}

fn builtin_info(descriptor: EngineDescriptor) -> BackendInfo {
    let name = friendly_engine_name(&descriptor.name);
    let engine = BackendEngine {
        id: descriptor.name.clone(),
        name: name.clone(),
        runtime: format!("{:?}", descriptor.runtime).to_ascii_lowercase(),
        description: format!("内置 {name} 推理引擎"),
        vendor: name.clone(),
        variant: "内置".into(),
        category: engine_category(&descriptor.name).into(),
        homepage: None,
        implementations: descriptor.implementations,
        formats: descriptor
            .formats
            .iter()
            .map(|format| format!("{format:?}").to_ascii_lowercase())
            .collect(),
        modalities: descriptor
            .modalities
            .iter()
            .map(|modality| format!("{modality:?}").to_ascii_lowercase())
            .collect(),
        devices: descriptor.devices,
    };
    BackendInfo::builtin(env!("CARGO_PKG_VERSION"), engine)
}

fn engine_category(name: &str) -> &'static str {
    if name == "whisper" {
        "audio"
    } else {
        "language"
    }
}

fn friendly_engine_name(name: &str) -> String {
    match name {
        "onnx" => "ONNX Runtime".into(),
        "candle" => "Candle".into(),
        "llama" => "llama.cpp".into(),
        "llama-mtmd" => "llama.cpp MTMD".into(),
        "whisper" => "whisper.cpp".into(),
        other => other.to_owned(),
    }
}

fn selected_backend(state: &AppState) -> Result<Option<(String, String)>, ApiError> {
    match state.backends.active() {
        Ok(Some(backend)) => Ok(Some((
            backend.manifest().id.clone(),
            backend.manifest().version.clone(),
        ))),
        Ok(None) => Ok(Some((
            BUILTIN_BACKEND_ID.to_owned(),
            env!("CARGO_PKG_VERSION").to_owned(),
        ))),
        Err(error) => {
            tracing::warn!(%error, "ignoring invalid active backend marker");
            Ok(Some((
                BUILTIN_BACKEND_ID.to_owned(),
                env!("CARGO_PKG_VERSION").to_owned(),
            )))
        }
    }
}

fn running_backend(state: &AppState) -> (String, String) {
    state.active_backend.clone().unwrap_or_else(|| {
        (
            BUILTIN_BACKEND_ID.to_owned(),
            env!("CARGO_PKG_VERSION").to_owned(),
        )
    })
}

fn backend_ref(id: &str, version: &str) -> Value {
    let kind = if id == BUILTIN_BACKEND_ID {
        BackendKind::Builtin
    } else {
        BackendKind::Sidecar
    };
    json!({"id": id, "version": version, "kind": kind})
}

fn latest_version(store: &BackendStore, id: &str) -> Result<String, ApiError> {
    let versions = store
        .installed()
        .map_err(|error| ApiError::internal(format!("read backend store: {error}")))?
        .into_iter()
        .filter(|backend| backend.manifest().id == id)
        .map(|backend| backend.manifest().version.clone())
        .collect::<Vec<_>>();
    versions
        .into_iter()
        .max_by_key(|version| version_key(version))
        .ok_or_else(|| {
            ApiError::not_found(
                "backend_not_found",
                format!("backend {id} is not installed"),
            )
        })
}

fn version_key(version: &str) -> Vec<u64> {
    version
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<u64>().ok())
        .collect()
}
