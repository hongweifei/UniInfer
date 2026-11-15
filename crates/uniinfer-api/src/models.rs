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
    State(state): State<AppState>,
    ApiJson(request): ApiJson<InspectRequest>,
) -> Result<Json<Value>, ApiError> {
    let path = sandboxed_path(&state, &request.path)?;
    let inspection = inspect(
        &path,
        &InspectOptions {
            id: request.id,
            ..InspectOptions::default()
        },
    )
    .map_err(ApiError::from)?;
    let route = state.runtime.probe_manifest(&inspection.manifest);
    let route = match route {
        Ok(route) => json!({
            "supported": true,
            "engine": route.engine_name,
            "placement": route.placement,
        }),
        Err(error) => json!({
            "supported": false,
            "code": error.code(),
            "message": error.to_string(),
        }),
    };
    Ok(Json(json!({
        "root": inspection.root,
        "manifest": inspection.manifest,
        "warnings": inspection.warnings,
        "route": route,
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
    let path = sandboxed_path(&state, &request.path)?;
    let inspection = inspect(
        &path,
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

/// Restrict model directory scans to the configuration root (or the process
/// working directory when no config file is attached).
fn sandboxed_path(
    state: &AppState,
    requested: &std::path::Path,
) -> Result<std::path::PathBuf, ApiError> {
    let root = state
        .config
        .as_ref()
        .map_or_else(std::env::current_dir, |handle| {
            std::env::current_dir().map(|cwd| cwd.join(handle.base_dir()))
        })
        .map_err(|error| ApiError::internal(format!("resolve sandbox root: {error}")))?;
    let root = root.canonicalize().map_err(|error| {
        ApiError::internal(format!(
            "canonicalize sandbox root {}: {error}",
            root.display()
        ))
    })?;
    let target = requested.canonicalize().map_err(|error| {
        ApiError::bad_request(
            "invalid_model_path",
            format!("cannot resolve {}: {error}", requested.display()),
        )
    })?;
    if !target.starts_with(&root) {
        return Err(ApiError::forbidden(
            "path_not_allowed",
            format!(
                "{} is outside the configured model root {}",
                target.display(),
                root.display()
            ),
        ));
    }
    Ok(target)
}
