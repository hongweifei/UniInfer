//! Read and replace the manifest backing a loaded model.

use axum::Json;
use axum::extract::{Path, State};
use serde_json::{Value, json};
use uniinfer_core::ModelManifest;

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::state::AppState;

pub(crate) async fn get(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let path = state
        .manifest_paths
        .read()
        .ok()
        .and_then(|paths| paths.get(&id).cloned())
        .ok_or_else(|| {
            ApiError::not_found("model_not_found", format!("model {id} is not loaded"))
        })?;
    let manifest = ModelManifest::load(&path)?;
    Ok(Json(json!({
        "id": id,
        "path": path,
        "manifest": manifest,
    })))
}

pub(crate) async fn put(
    Path(id): Path<String>,
    State(state): State<AppState>,
    ApiJson(manifest): ApiJson<ModelManifest>,
) -> Result<Json<Value>, ApiError> {
    let path = state
        .manifest_paths
        .read()
        .ok()
        .and_then(|paths| paths.get(&id).cloned())
        .ok_or_else(|| {
            ApiError::not_found("model_not_found", format!("model {id} is not loaded"))
        })?;
    if manifest.id != id {
        return Err(ApiError::bad_request(
            "manifest_id_mismatch",
            format!("manifest id {} does not match {id}", manifest.id),
        ));
    }
    manifest.save(&path)?;
    state.update_manifest(&id, manifest)?;
    Ok(Json(json!({
        "saved": true,
        "reloaded": true,
        "restart_required": false,
        "path": path,
    })))
}
