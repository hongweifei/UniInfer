use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{ConfigError, Result};
use crate::format::ConfigFormat;

pub const SCHEMA_VERSION: u32 = 1;

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
        let config: Self = ConfigFormat::for_path(path).parse(&raw, path)?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let raw = ConfigFormat::for_path(path).serialize(self)?;
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
        if !self.server.addr.ip().is_loopback() && self.server.token.is_none() {
            return Err(ConfigError::Invalid(
                "server.token is required when server.addr is not loopback".into(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_toml() {
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
    fn non_loopback_serve_requires_token() {
        let config = UniInferConfig {
            server: ServerConfig {
                addr: "0.0.0.0:8080".parse().expect("addr"),
                ..ServerConfig::default()
            },
            ..UniInferConfig::default()
        };
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
        let with_token = UniInferConfig {
            server: ServerConfig {
                addr: "0.0.0.0:8080".parse().expect("addr"),
                token: Some("secret".into()),
                ..ServerConfig::default()
            },
            ..UniInferConfig::default()
        };
        assert!(with_token.validate().is_ok());
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
