//! Discovery and lifecycle for downloadable sidecar inference backends.
//!
//! A backend package is a directory containing `backend.toml` and the worker
//! executable it names. Packages are versioned on disk and every installed
//! variant is exposed to model manifests; there is no global active package.

use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use uniinfer_core::DeviceClass;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("read backend store: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse backend manifest: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("serialize backend manifest: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("invalid backend package: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// Stable identifier for the in-process backend that ships with UniInfer.
pub const BUILTIN_BACKEND_ID: &str = "builtin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Builtin,
    Sidecar,
}

/// One inference engine exposed by a backend package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendEngine {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub description: String,
    /// Upstream project/vendor shown as the card title, e.g. `llama.cpp`.
    #[serde(default)]
    pub vendor: String,
    /// Hardware/provider variant shown as a row, e.g. `Vulkan`, `CPU`.
    #[serde(default)]
    pub variant: String,
    /// Primary UI category: `language`, `audio`, or `image`.
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub implementations: Vec<String>,
    #[serde(default)]
    pub formats: Vec<String>,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub devices: Vec<DeviceClass>,
}

/// Uniform description used by the API and GUI for both backend kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub platform: String,
    pub devices: Vec<DeviceClass>,
    pub kind: BackendKind,
    pub size_bytes: u64,
    #[serde(default)]
    pub engines: Vec<BackendEngine>,
}

impl BackendInfo {
    #[must_use]
    pub fn builtin(version: impl Into<String>, engine: BackendEngine) -> Self {
        Self {
            id: format!("builtin:{}", engine.id),
            name: engine.name.clone(),
            version: version.into(),
            description: engine.description.clone(),
            platform: std::env::consts::OS.into(),
            devices: engine.devices.clone(),
            kind: BackendKind::Builtin,
            size_bytes: 0,
            engines: vec![engine],
        }
    }
}

/// Runtime composition: built-ins, one sidecar, or every installed sidecar.
#[derive(Debug, Clone)]
pub enum BackendRuntime {
    Builtin,
    Sidecar(BackendLaunch),
    Sidecars(Vec<BackendLaunch>),
}

/// A downloadable backend package.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    /// Informational platform label such as `windows-x86_64`.
    #[serde(default)]
    pub platform: String,
    /// Worker/server executable path relative to the package directory.
    pub worker: String,
    #[serde(default)]
    pub launcher: BackendLauncher,
    #[serde(default)]
    pub devices: Vec<DeviceClass>,
    #[serde(default)]
    pub engines: Vec<BackendEngine>,
    #[serde(default)]
    pub files: Vec<BackendFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendFile {
    pub path: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendCatalog {
    #[serde(default)]
    pub backends: Vec<BackendManifest>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// An installed package discovered on disk.
#[derive(Debug, Clone)]
pub struct InstalledBackend {
    dir: PathBuf,
    manifest: BackendManifest,
}

/// How the selected distribution is launched by the runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendLauncher {
    /// UniInfer's tensor worker protocol.
    #[default]
    Worker,
    /// Official llama.cpp `llama-server`.
    LlamaServer,
    /// Official stable-diffusion.cpp `sd-server`.
    StableDiffusionServer,
}

/// The subset needed by the engine registry to launch a sidecar.
#[derive(Debug, Clone)]
pub struct BackendLaunch {
    pub id: String,
    pub version: String,
    pub family: String,
    pub variant: String,
    pub launcher: BackendLauncher,
    pub worker: PathBuf,
    pub devices: Vec<DeviceClass>,
}

impl InstalledBackend {
    #[must_use]
    pub fn info(&self) -> BackendInfo {
        BackendInfo {
            id: self.manifest.id.clone(),
            name: self.manifest.name.clone(),
            version: self.manifest.version.clone(),
            description: self.manifest.description.clone(),
            platform: self.manifest.platform.clone(),
            devices: self.manifest.devices.clone(),
            kind: BackendKind::Sidecar,
            size_bytes: self.size_bytes(),
            engines: self.manifest.engines.clone(),
        }
    }

    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    #[must_use]
    pub fn manifest(&self) -> &BackendManifest {
        &self.manifest
    }

    #[must_use]
    pub fn worker_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.worker)
    }

    pub fn launch(&self) -> Result<BackendLaunch> {
        let worker = self.worker_path();
        if !worker.is_file() {
            return Err(BackendError::Invalid(format!(
                "{} worker is missing: {}",
                self.manifest.id,
                worker.display()
            )));
        }
        let variant = self
            .manifest
            .engines
            .first()
            .map(|engine| engine.variant.trim())
            .filter(|variant| !variant.is_empty())
            .unwrap_or("auto")
            .to_owned();
        Ok(BackendLaunch {
            id: self.manifest.id.clone(),
            version: self.manifest.version.clone(),
            family: self.manifest.name.trim().to_owned(),
            variant,
            launcher: self.manifest.launcher,
            worker,
            devices: self.manifest.devices.clone(),
        })
    }

    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.manifest.files.iter().map(|file| file.size).sum()
    }
}

/// Filesystem-backed backend package store.
#[derive(Debug, Clone)]
pub struct BackendStore {
    root: PathBuf,
}

impl BackendStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    /// Return every valid installed package, newest package directory order
    pub fn installed(&self) -> Result<Vec<InstalledBackend>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut packages = Vec::new();
        for id_entry in fs::read_dir(&self.root)? {
            let id_entry = id_entry?;
            if !id_entry.file_type()?.is_dir() {
                continue;
            }
            for version_entry in fs::read_dir(id_entry.path())? {
                let version_entry = version_entry?;
                if !version_entry.file_type()?.is_dir() {
                    continue;
                }
                match load_installed(&version_entry.path()) {
                    Ok(package) => packages.push(package),
                    Err(error) => {
                        tracing::warn!(path = %version_entry.path().display(), %error, "ignoring invalid backend package");
                    }
                }
            }
        }
        packages.sort_by(|a, b| {
            a.manifest
                .id
                .cmp(&b.manifest.id)
                .then_with(|| a.manifest.version.cmp(&b.manifest.version))
        });
        Ok(packages)
    }

    /// Remove every installed version of a backend.
    pub fn remove(&self, id: &str) -> Result<bool> {
        validate_component(id, "backend id")?;
        let dir = self.root.join(id);
        let removed = if dir.is_dir() {
            fs::remove_dir_all(&dir)?;
            true
        } else {
            false
        };
        Ok(removed)
    }

    /// Directory used while downloading or copying a package.
    #[must_use]
    pub fn package_dir(&self, id: &str, version: &str) -> PathBuf {
        self.root.join(id).join(version)
    }

    /// Validate that a relative package path cannot escape its directory.
    pub fn safe_relative(path: &str) -> bool {
        let path = Path::new(path);
        !path.as_os_str().is_empty()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
    }

    /// Persist a manifest inside a package directory.
    pub fn write_manifest(&self, dir: &Path, manifest: &BackendManifest) -> Result<()> {
        fs::create_dir_all(dir)?;
        let raw = toml::to_string_pretty(manifest)?;
        write_atomic(&dir.join("backend.toml"), raw.as_bytes())?;
        Ok(())
    }
}

fn load_installed(dir: &Path) -> Result<InstalledBackend> {
    let manifest_path = dir.join("backend.toml");
    let raw = fs::read_to_string(&manifest_path)?;
    let manifest: BackendManifest = toml::from_str(&raw)?;
    validate_manifest(&manifest)?;
    if !dir.join(&manifest.worker).is_file() {
        return Err(BackendError::Invalid(format!(
            "{} worker is missing",
            manifest.id
        )));
    }
    Ok(InstalledBackend {
        dir: dir.to_path_buf(),
        manifest,
    })
}

pub fn validate_manifest(manifest: &BackendManifest) -> Result<()> {
    validate_component(&manifest.id, "backend id")?;
    validate_component(&manifest.version, "backend version")?;
    if manifest.name.trim().is_empty() {
        return Err(BackendError::Invalid("backend name is empty".into()));
    }
    if !BackendStore::safe_relative(&manifest.worker) {
        return Err(BackendError::Invalid(format!(
            "unsafe worker path: {}",
            manifest.worker
        )));
    }
    if manifest.engines.is_empty() {
        return Err(BackendError::Invalid(
            "backend must declare at least one [[engines]] entry".into(),
        ));
    }
    for engine in &manifest.engines {
        if engine.id.trim().is_empty() || engine.name.trim().is_empty() {
            return Err(BackendError::Invalid(
                "backend engine id and name are required".into(),
            ));
        }
    }
    for file in &manifest.files {
        if !BackendStore::safe_relative(&file.path) {
            return Err(BackendError::Invalid(format!(
                "unsafe package file path: {}",
                file.path
            )));
        }
    }
    Ok(())
}

fn validate_component(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains(['/', '\\'])
        || value.contains('\0')
    {
        return Err(BackendError::Invalid(format!("invalid {label}: {value}")));
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    // Windows cannot rename over an existing file.
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(root: &Path, id: &str, version: &str) -> InstalledBackend {
        let dir = root.join(id).join(version);
        fs::create_dir_all(dir.join("bin")).expect("dir");
        fs::write(dir.join("bin/worker.exe"), b"worker").expect("worker");
        let manifest = BackendManifest {
            id: id.into(),
            name: "Test backend".into(),
            version: version.into(),
            description: String::new(),
            homepage: None,
            license: None,
            platform: "test".into(),
            worker: "bin/worker.exe".into(),
            launcher: BackendLauncher::Worker,
            devices: vec![DeviceClass::Gpu],
            engines: vec![BackendEngine {
                id: "test-vulkan".into(),
                name: "Test Vulkan".into(),
                runtime: "test".into(),
                description: "test engine".into(),
                vendor: "Test".into(),
                variant: "CPU".into(),
                category: "language".into(),
                homepage: None,
                implementations: vec!["test.graph".into()],
                formats: vec!["gguf".into()],
                modalities: vec!["text".into()],
                devices: vec![DeviceClass::Gpu],
            }],
            files: Vec::new(),
        };
        BackendStore::new(root)
            .write_manifest(&dir, &manifest)
            .expect("manifest");
        load_installed(&dir).expect("installed")
    }

    #[test]
    fn discovers_and_removes_packages() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = BackendStore::new(temp.path());
        package(temp.path(), "llama-vulkan", "1.0.0");
        package(temp.path(), "llama-vulkan", "1.1.0");
        package(temp.path(), "onnx-cuda", "2.0.0");
        assert_eq!(store.installed().expect("installed").len(), 3);
        let installed = store.installed().expect("installed");
        assert!(installed.iter().all(|package| package.launch().is_ok()));
        assert!(store.remove("llama-vulkan").expect("remove"));
        assert_eq!(store.installed().expect("installed").len(), 1);
    }

    #[test]
    fn rejects_path_traversal() {
        assert!(!BackendStore::safe_relative("../worker"));
        assert!(!BackendStore::safe_relative("/worker"));
        assert!(!BackendStore::safe_relative("bin/../../worker"));
        assert!(BackendStore::safe_relative("bin/worker"));
    }
}
