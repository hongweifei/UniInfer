//! Shared parsing and path checks for adapter-owned `[audio.options]`.

use std::path::{Component, Path, PathBuf};

use serde::de::DeserializeOwned;
use uniinfer_core::{AudioConfig, EngineError, Result};

pub(crate) fn parse<T: DeserializeOwned>(audio: &AudioConfig) -> Result<T> {
    let options = audio.options.clone().ok_or_else(|| {
        EngineError::ManifestInvalid("audio.options is required for this architecture".into())
    })?;
    options
        .try_into()
        .map_err(|error| EngineError::ManifestInvalid(format!("audio.options: {error}")))
}

pub(crate) fn path(base: &Path, path: &Path, field: &str) -> Result<PathBuf> {
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(EngineError::ManifestInvalid(format!(
            "audio.options.{field} must stay inside the model directory: {}",
            path.display()
        )));
    }
    Ok(base.join(path))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn rejects_parent_paths() {
        let error = path(
            Path::new("models"),
            Path::new("../weights.onnx"),
            "graph_file",
        )
        .expect_err("must reject");
        assert!(error.to_string().contains("model directory"));
    }

    #[test]
    fn resolves_relative_paths() {
        let path = path(Path::new("models"), Path::new("weights.onnx"), "graph_file")
            .expect("relative path");
        assert!(path.ends_with("weights.onnx"));
    }
}
