use std::net::SocketAddr;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ConfigError, Result};
use crate::format::ConfigFormat;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniInferConfig {
    #[serde(default)]
    pub server: ServerConfig,
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
    /// Maximum models kept loaded per category; `0` means unbounded.
    #[serde(default)]
    pub max_loaded_models: usize,
    /// Per-request generation timeout; `0` disables the deadline.
    #[serde(default)]
    pub request_timeout_ms: u64,
    /// Model load timeout; `0` disables the deadline.
    #[serde(default)]
    pub model_load_timeout_ms: u64,
    /// Route text models through `uniinfer-worker` child processes.
    #[serde(default)]
    pub isolate_models: bool,
    /// Memory held back from GPU scheduling, in MiB.
    #[serde(default)]
    pub vram_reserve_mb: u64,
    /// Memory held back from host scheduling, in MiB.
    #[serde(default)]
    pub host_reserve_mb: u64,
    /// Directory containing downloadable sidecar backend packages.
    /// Relative paths resolve against the config file directory.
    #[serde(default)]
    pub backend_dir: Option<std::path::PathBuf>,
}

fn default_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 11435))
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
            max_loaded_models: 0,
            request_timeout_ms: 0,
            model_load_timeout_ms: 0,
            isolate_models: false,
            vram_reserve_mb: 0,
            host_reserve_mb: 0,
            backend_dir: None,
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
        Ok(())
    }

    pub fn validate_for_serve(&self) -> Result<()> {
        self.validate()
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
        };
        config.save(&path).expect("save");
        let loaded = UniInferConfig::load(&path).expect("load");
        assert_eq!(loaded.server.addr.port(), 9090);
    }

    #[test]
    fn non_loopback_serve_requires_token() {
        let config = UniInferConfig {
            server: ServerConfig {
                addr: "0.0.0.0:8080".parse().expect("addr"),
                ..ServerConfig::default()
            },
        };
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
        let with_token = UniInferConfig {
            server: ServerConfig {
                addr: "0.0.0.0:8080".parse().expect("addr"),
                token: Some("secret".into()),
                ..ServerConfig::default()
            },
        };
        assert!(with_token.validate().is_ok());
    }
}
