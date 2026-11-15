use std::fs;
use std::path::PathBuf;

use uniinfer_core::ModelManifest;

use crate::error::{InspectError, Result};

#[derive(Debug, Clone)]
pub struct Inspection {
    pub root: PathBuf,
    pub manifest: ModelManifest,
    pub warnings: Vec<String>,
}

impl Inspection {
    pub fn to_toml(&self) -> Result<String> {
        self.manifest
            .validate()
            .map_err(|error| InspectError::Manifest(error.to_string()))?;
        toml::to_string_pretty(&self.manifest)
            .map_err(|error| InspectError::Manifest(error.to_string()))
    }

    /// Write `manifest.toml` next to the model files.
    pub fn write_manifest(&self) -> Result<PathBuf> {
        let path = self.root.join("manifest.toml");
        fs::write(&path, self.to_toml()?).map_err(|source| InspectError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }
}
