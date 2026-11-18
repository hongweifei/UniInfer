//! Unified list and selection of built-in and sidecar inference backends.

use axum::Json;
use axum::extract::{Path, State};
use serde_json::{Value, json};
use uniinfer_backend_store::{BUILTIN_BACKEND_ID, BackendEngine, BackendInfo, InstalledBackend};
use uniinfer_registry::EngineDescriptor;

use crate::error::ApiError;
use crate::state::AppState;

pub(crate) async fn list(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    state
        .sync_backends()
        .map_err(|error| ApiError::internal(format!("sync backends: {error}")))?;
    let installed = state
        .backends
        .installed()
        .map_err(|error| ApiError::internal(format!("read backend store: {error}")))?;
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
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(json!({
        "directory": state.backends.root(),
        "backends": data,
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
    let launched = state
        .backends
        .installed()
        .map_err(|error| ApiError::internal(format!("read backend store: {error}")))?
        .into_iter()
        .filter(|backend| backend.manifest().id == id)
        .filter_map(|backend| backend.launch().ok())
        .collect::<Vec<_>>();
    let removed = state
        .backends
        .remove(&id)
        .map_err(|error| ApiError::bad_request("backend_remove_failed", error.to_string()))?;
    for launch in launched {
        state
            .runtime
            .unregister_sidecar(&launch.family, &launch.variant);
    }
    if let Err(error) = state.sync_backends() {
        tracing::warn!(%error, "re-register remaining backends after removal");
    }
    Ok(Json(json!({
        "id": id,
        "removed": removed,
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
        variants: builtin_variants(&descriptor.name),
    };
    BackendInfo::builtin(env!("CARGO_PKG_VERSION"), engine)
}

fn builtin_variants(name: &str) -> Vec<String> {
    if name != "onnx" {
        return Vec::new();
    }
    let mut variants = vec!["CPU".to_owned()];
    if cfg!(feature = "ep-cuda") {
        variants.push("CUDA".into());
    }
    if cfg!(feature = "ep-tensorrt") {
        variants.push("TensorRT".into());
    }
    if cfg!(feature = "ep-directml") {
        variants.push("DirectML".into());
    }
    if cfg!(feature = "ep-openvino") {
        variants.push("OpenVINO".into());
    }
    if cfg!(feature = "ep-rocm") {
        variants.push("ROCm".into());
    }
    if cfg!(feature = "ep-coreml") {
        variants.push("CoreML".into());
    }
    if cfg!(feature = "ep-qnn") {
        variants.push("QNN".into());
    }
    if cfg!(feature = "ep-nnapi") {
        variants.push("NNAPI".into());
    }
    if cfg!(feature = "ep-vitis") {
        variants.push("Vitis AI".into());
    }
    if cfg!(feature = "ep-cann") {
        variants.push("CANN".into());
    }
    if cfg!(feature = "ep-onednn") {
        variants.push("OneDNN".into());
    }
    if cfg!(feature = "ep-xnnpack") {
        variants.push("XNNPACK".into());
    }
    if cfg!(feature = "ep-acl") {
        variants.push("ACL".into());
    }
    if cfg!(feature = "ep-migraphx") {
        variants.push("MIGraphX".into());
    }
    if cfg!(feature = "ep-nvrtx") {
        variants.push("NVRTX".into());
    }
    if cfg!(feature = "ep-webgpu") {
        variants.push("WebGPU".into());
    }
    variants
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
