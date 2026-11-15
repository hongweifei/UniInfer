//! Typed application configuration shared by the CLI, API and future GUI.
//!
//! The schema contains only durable user choices. Runtime objects (loaded
//! models, ORT sessions, devices) are deliberately not serialized here.

use std::collections::HashSet;
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read config {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot write config {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse config {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("cannot serialize config: {0}")]
    Serialize(String),
    #[error("invalid config: {0}")]
    Invalid(String),
    #[error("config handle has no file path")]
    NoPath,
    #[error("config lock poisoned")]
    LockPoisoned,
}

pub type Result<T, E = ConfigError> = std::result::Result<T, E>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    Toml,
    Json,
}

impl ConfigFormat {
    fn for_path(path: &Path) -> Self {
        match path.extension().and_then(|value| value.to_str()) {
            Some("json") => Self::Json,
            _ => Self::Toml,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniInferConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_addr")]
    pub addr: SocketAddr,
    #[serde(default = "default_queue_timeout_ms")]
    pub queue_timeout_ms: u64,
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    #[serde(default)]
    pub token: Option<String>,
}

fn default_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 8080))
}

fn default_queue_timeout_ms() -> u64 {
    30_000
}

fn default_max_body_bytes() -> usize {
    32 * 1024 * 1024
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: default_addr(),
            queue_timeout_ms: default_queue_timeout_ms(),
            max_body_bytes: default_max_body_bytes(),
            token: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub manifest: PathBuf,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_enabled() -> bool {
    true
}

impl ModelEntry {
    #[must_use]
    pub fn new(manifest: PathBuf) -> Self {
        Self {
            manifest,
            enabled: true,
            name: None,
            tags: Vec::new(),
        }
    }

    #[must_use]
    pub fn resolved_manifest(&self, base: &Path) -> PathBuf {
        if self.manifest.is_absolute() {
            self.manifest.clone()
        } else {
            base.join(&self.manifest)
        }
    }
}

impl Default for UniInferConfig {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            server: ServerConfig::default(),
            models: Vec::new(),
        }
    }
}

impl UniInferConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Self = match ConfigFormat::for_path(path) {
            ConfigFormat::Toml => toml::from_str(&raw).map_err(|error| ConfigError::Parse {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?,
            ConfigFormat::Json => {
                serde_json::from_str(&raw).map_err(|error| ConfigError::Parse {
                    path: path.to_path_buf(),
                    message: error.to_string(),
                })?
            }
        };
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let raw = match ConfigFormat::for_path(path) {
            ConfigFormat::Toml => toml::to_string_pretty(self)
                .map_err(|error| ConfigError::Serialize(error.to_string()))?,
            ConfigFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|error| ConfigError::Serialize(error.to_string()))?,
        };
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|source| ConfigError::Write {
                path: path.to_path_buf(),
                source,
            })?;
        }
        std::fs::write(path, raw).map_err(|source| ConfigError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ConfigError::Invalid(format!(
                "schema_version must be {SCHEMA_VERSION}, got {}",
                self.schema_version
            )));
        }
        if self.server.queue_timeout_ms == 0 {
            return Err(ConfigError::Invalid(
                "server.queue_timeout_ms must be positive".into(),
            ));
        }
        if self.server.max_body_bytes == 0 {
            return Err(ConfigError::Invalid(
                "server.max_body_bytes must be positive".into(),
            ));
        }
        if self.server.token.as_deref() == Some("") {
            return Err(ConfigError::Invalid(
                "server.token must not be empty; omit it to disable auth".into(),
            ));
        }
        let mut manifests = HashSet::new();
        for model in &self.models {
            if model.manifest.as_os_str().is_empty() {
                return Err(ConfigError::Invalid(
                    "models[].manifest must not be empty".into(),
                ));
            }
            if !manifests.insert(model.manifest.clone()) {
                return Err(ConfigError::Invalid(format!(
                    "duplicate model manifest {}",
                    model.manifest.display()
                )));
            }
        }
        Ok(())
    }

    pub fn validate_for_serve(&self) -> Result<()> {
        self.validate()?;
        if self.enabled_models().next().is_none() {
            return Err(ConfigError::Invalid(
                "at least one enabled model is required to serve".into(),
            ));
        }
        Ok(())
    }

    pub fn enabled_models(&self) -> impl Iterator<Item = &ModelEntry> {
        self.models.iter().filter(|model| model.enabled)
    }

    pub fn apply_env(&mut self) -> Result<()> {
        if let Ok(value) = std::env::var("UNIINFER_ADDR") {
            self.server.addr = value
                .parse()
                .map_err(|error| ConfigError::Invalid(format!("UNIINFER_ADDR: {error}")))?;
        }
        if let Ok(value) = std::env::var("UNIINFER_QUEUE_TIMEOUT_MS") {
            self.server.queue_timeout_ms = value.parse().map_err(|error| {
                ConfigError::Invalid(format!("UNIINFER_QUEUE_TIMEOUT_MS: {error}"))
            })?;
        }
        if let Ok(value) = std::env::var("UNIINFER_MAX_BODY_BYTES") {
            self.server.max_body_bytes = value.parse().map_err(|error| {
                ConfigError::Invalid(format!("UNIINFER_MAX_BODY_BYTES: {error}"))
            })?;
        }
        if let Ok(value) = std::env::var("UNIINFER_TOKEN") {
            self.server.token = if value.is_empty() { None } else { Some(value) };
        }
        self.validate()
    }
}

/// Thread-safe current config plus its optional source file.
pub struct ConfigHandle {
    path: Option<PathBuf>,
    config: RwLock<UniInferConfig>,
}

impl fmt::Debug for ConfigHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConfigHandle")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl ConfigHandle {
    #[must_use]
    pub fn new(config: UniInferConfig, path: Option<PathBuf>) -> Self {
        Self {
            path,
            config: RwLock::new(config),
        }
    }

    pub fn load(path: PathBuf) -> Result<Self> {
        let config = UniInferConfig::load(&path)?;
        Ok(Self::new(config, Some(path)))
    }

    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    #[must_use]
    pub fn base_dir(&self) -> PathBuf {
        self.path
            .as_deref()
            .and_then(Path::parent)
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    }

    pub fn snapshot(&self) -> Result<UniInferConfig> {
        self.config
            .read()
            .map(|config| config.clone())
            .map_err(|_| ConfigError::LockPoisoned)
    }

    /// Validate, persist to the source file, then publish the new snapshot.
    pub fn replace(&self, next: UniInferConfig) -> Result<()> {
        next.validate()?;
        let path = self.path.as_deref().ok_or(ConfigError::NoPath)?;
        next.save(path)?;
        *self.config.write().map_err(|_| ConfigError::LockPoisoned)? = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_toml_through_handle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("uniinfer.toml");
        let config = UniInferConfig {
            server: ServerConfig {
                addr: "127.0.0.1:9090".parse().expect("addr"),
                ..ServerConfig::default()
            },
            models: vec![ModelEntry::new("models/tiny.toml".into())],
            ..UniInferConfig::default()
        };
        config.save(&path).expect("save");
        let loaded = UniInferConfig::load(&path).expect("load");
        assert_eq!(loaded.server.addr.port(), 9090);
        assert_eq!(loaded.models.len(), 1);
    }

    #[test]
    fn rejects_duplicate_manifest_paths() {
        let config = UniInferConfig {
            models: vec![
                ModelEntry::new("a.toml".into()),
                ModelEntry::new("a.toml".into()),
            ],
            ..UniInferConfig::default()
        };
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
    }
}
