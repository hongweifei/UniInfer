//! Discovery and lifecycle for downloadable sidecar inference backends.
//!
//! A backend package is a directory containing `backend.toml` and the worker
//! executable it names. Packages are versioned on disk; only one package is
//! active at a time, and activation takes effect when the API runtime restarts.

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
    /// Worker executable path relative to the package directory.
    pub worker: String,
    #[serde(default)]
    pub devices: Vec<DeviceClass>,
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
}

/// An installed package discovered on disk.
#[derive(Debug, Clone)]
pub struct InstalledBackend {
    dir: PathBuf,
    manifest: BackendManifest,
}

/// The subset needed by the engine registry to launch a sidecar.
#[derive(Debug, Clone)]
pub struct BackendLaunch {
    pub id: String,
    pub version: String,
    pub worker: PathBuf,
    pub devices: Vec<DeviceClass>,
}

impl InstalledBackend {
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
        Ok(BackendLaunch {
            id: self.manifest.id.clone(),
            version: self.manifest.version.clone(),
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
    /// independent of the active marker.
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

    /// Read the active marker and resolve it to an installed package.
    pub fn active(&self) -> Result<Option<InstalledBackend>> {
        let marker = self.root.join("active");
        let raw = match fs::read_to_string(&marker) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut parts = raw.trim().splitn(2, '/');
        let Some(id) = parts.next().filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        let Some(version) = parts.next().filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        let dir = self.root.join(id).join(version);
        if !dir.is_dir() {
            return Ok(None);
        }
        Ok(Some(load_installed(&dir)?))
    }

    /// Make one package active. The runtime must restart before it is used.
    pub fn set_active(&self, id: &str, version: &str) -> Result<()> {
        validate_component(id, "backend id")?;
        validate_component(version, "backend version")?;
        let dir = self.root.join(id).join(version);
        if !dir.is_dir() {
            return Err(BackendError::Invalid(format!(
                "backend {id} {version} is not installed"
            )));
        }
        load_installed(&dir)?;
        self.ensure()?;
        write_atomic(
            &self.root.join("active"),
            format!("{id}/{version}\n").as_bytes(),
        )?;
        Ok(())
    }

    pub fn clear_active(&self) -> Result<()> {
        self.ensure()?;
        match fs::remove_file(self.root.join("active")) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    /// Remove all versions of a backend and clear the active marker if needed.
    pub fn remove(&self, id: &str) -> Result<bool> {
        validate_component(id, "backend id")?;
        let dir = self.root.join(id);
        let marker = self.root.join("active");
        let was_active = fs::read_to_string(&marker)
            .ok()
            .is_some_and(|raw| raw.trim().split('/').next() == Some(id));
        let removed = if dir.is_dir() {
            fs::remove_dir_all(&dir)?;
            true
        } else {
            false
        };
        if was_active {
            self.clear_active()?;
        }
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
            devices: vec![DeviceClass::Gpu],
            files: Vec::new(),
        };
        BackendStore::new(root)
            .write_manifest(&dir, &manifest)
            .expect("manifest");
        load_installed(&dir).expect("installed")
    }

    #[test]
    fn discovers_and_activates_packages() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = BackendStore::new(temp.path());
        package(temp.path(), "llama-vulkan", "1.0.0");
        package(temp.path(), "llama-vulkan", "1.1.0");
        package(temp.path(), "onnx-cuda", "2.0.0");
        assert_eq!(store.installed().expect("installed").len(), 3);
        store.set_active("llama-vulkan", "1.0.0").expect("active");
        store
            .set_active("llama-vulkan", "1.1.0")
            .expect("active replacement");
        let active = store.active().expect("active").expect("present");
        assert_eq!(active.manifest().version, "1.1.0");
        assert!(active.launch().is_ok());
        assert!(store.remove("llama-vulkan").expect("remove"));
        assert!(store.active().expect("active").is_none());
        assert!(!temp.path().join("active").exists());
    }

    #[test]
    fn rejects_path_traversal() {
        assert!(!BackendStore::safe_relative("../worker"));
        assert!(!BackendStore::safe_relative("/worker"));
        assert!(!BackendStore::safe_relative("bin/../../worker"));
        assert!(BackendStore::safe_relative("bin/worker"));
    }
}
