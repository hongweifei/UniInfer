//! Installed sidecar backend packages.

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;
use serde_json::{Value, json};
use uniinfer_backend_store::BackendStore;

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
    let active = match state.backends.active() {
        Ok(active) => active.map(|backend| {
            (
                backend.manifest().id.clone(),
                backend.manifest().version.clone(),
            )
        }),
        Err(error) => {
            tracing::warn!(%error, "ignoring invalid active backend marker");
            None
        }
    };
    let running = state.active_backend.clone();
    let data = installed
        .iter()
        .map(|backend| {
            let manifest = backend.manifest();
            let is_active = active
                .as_ref()
                .is_some_and(|(id, version)| id == &manifest.id && version == &manifest.version);
            json!({
                "id": manifest.id,
                "name": manifest.name,
                "version": manifest.version,
                "description": manifest.description,
                "homepage": manifest.homepage,
                "license": manifest.license,
                "platform": manifest.platform,
                "devices": manifest.devices,
                "worker": manifest.worker,
                "size_bytes": backend.size_bytes(),
                "active": is_active,
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "active": active.map(|(id, version)| json!({"id": id, "version": version})),
        "running": running.map(|(id, version)| json!({"id": id, "version": version})),
        "directory": state.backends.root(),
        "catalog_url": state.backend_catalog_url,
        "installed": data,
    })))
}

pub(crate) async fn activate(
    Path(id): Path<String>,
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ActivateRequest>,
) -> Result<Json<Value>, ApiError> {
    let version = match request.version {
        Some(version) => version,
        None => latest_version(&state.backends, &id)?,
    };
    state
        .backends
        .set_active(&id, &version)
        .map_err(|error| ApiError::bad_request("backend_activate_failed", error.to_string()))?;
    Ok(Json(json!({
        "active": {"id": id, "version": version},
        "restart_required": true,
    })))
}

pub(crate) async fn remove(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
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

pub(crate) async fn deactivate(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    state
        .backends
        .clear_active()
        .map_err(|error| ApiError::internal(format!("clear active backend: {error}")))?;
    Ok(Json(json!({"active": null, "restart_required": true})))
}

fn version_key(version: &str) -> Vec<u64> {
    version
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<u64>().ok())
        .collect()
}
