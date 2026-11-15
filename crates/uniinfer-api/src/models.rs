//! Model discovery/import endpoints for the future GUI.

use axum::Json;
use axum::extract::State;
use serde::Deserialize;
use serde_json::{Value, json};
use uniinfer_config::ModelEntry;
use uniinfer_model_inspect::{InspectOptions, inspect};

use crate::error::ApiError;
use crate::extract::ApiJson;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub(crate) struct InspectRequest {
    path: std::path::PathBuf,
    #[serde(default)]
    id: Option<String>,
}

pub(crate) async fn inspect_model(
    ApiJson(request): ApiJson<InspectRequest>,
) -> Result<Json<Value>, ApiError> {
    let inspection = inspect(
        &request.path,
        &InspectOptions {
            id: request.id,
            ..InspectOptions::default()
        },
    )
    .map_err(ApiError::from)?;
    Ok(Json(json!({
        "root": inspection.root,
        "manifest": inspection.manifest,
        "warnings": inspection.warnings,
    })))
}

pub(crate) async fn import_model(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<InspectRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = state
        .config
        .as_ref()
        .ok_or_else(|| ApiError::not_found("config_not_available", "server has no config file"))?;
    let inspection = inspect(
        &request.path,
        &InspectOptions {
            id: request.id,
            ..InspectOptions::default()
        },
    )
    .map_err(ApiError::from)?;
    let manifest_path = inspection.write_manifest().map_err(ApiError::from)?;
    let base = handle.base_dir();
    let relative = manifest_path
        .strip_prefix(&base)
        .map_or(manifest_path.clone(), std::path::Path::to_path_buf);
    let mut config = handle.snapshot().map_err(ApiError::from)?;
    if !config.models.iter().any(|model| model.manifest == relative) {
        config.models.push(ModelEntry::new(relative.clone()));
    }
    config.validate().map_err(ApiError::from)?;
    handle.replace(config).map_err(ApiError::from)?;
    Ok(Json(json!({
        "manifest": manifest_path,
        "config": handle.path(),
        "restart_required": true,
    })))
}
