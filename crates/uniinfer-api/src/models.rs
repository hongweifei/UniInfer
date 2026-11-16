//! Model discovery/import endpoints for the future GUI.

use axum::Json;
use axum::extract::State;
use serde::Deserialize;
use serde_json::{Value, json};
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
    state
        .add_manifest(manifest_path.clone(), inspection.manifest)
        .map_err(ApiError::from)?;
    state.refresh_catalog().map_err(ApiError::from)?;
    Ok(Json(json!({
        "manifest": manifest_path,
        "restart_required": false,
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
