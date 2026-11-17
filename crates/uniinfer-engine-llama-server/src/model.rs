//! Model path selection and offload parameters for `llama-server`.

use std::path::PathBuf;

use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, Result};

pub(crate) fn model_path(spec: &EngineLoadSpec) -> Result<PathBuf> {
    let mut candidates = spec
        .manifest
        .files
        .iter()
        .filter(|file| {
            file.path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter(|file| {
            !file
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.to_ascii_lowercase().contains("mmproj"))
        })
        .map(|file| spec.base_dir.join(&file.path))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .max_by_key(|path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len())
                .unwrap_or(0)
        })
        .ok_or_else(|| EngineError::ManifestInvalid("no GGUF model file for llama-server".into()))
}

pub(crate) fn model_bytes(spec: &EngineLoadSpec) -> u64 {
    spec.manifest
        .files
        .iter()
        .filter(|file| {
            file.path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter_map(|file| std::fs::metadata(spec.base_dir.join(&file.path)).ok())
        .map(|metadata| metadata.len())
        .sum()
}

pub(crate) fn offload_layers(spec: &EngineLoadSpec) -> u32 {
    let gpu = spec
        .placement
        .devices
        .iter()
        .any(|device| device.class == DeviceClass::Gpu);
    if !gpu {
        return 0;
    }
    if spec.placement.auto_offload {
        return 999;
    }
    u32::try_from(spec.placement.offload_layers).unwrap_or(u32::MAX)
}
