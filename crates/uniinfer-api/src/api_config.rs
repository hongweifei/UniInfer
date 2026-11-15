use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_config::{ConfigError, ConfigHandle};

/// Server options.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub manifests: Vec<PathBuf>,
    pub queue_timeout: Duration,
    pub token: Option<String>,
    pub max_body_bytes: usize,
    pub max_loaded_models: usize,
    pub config: Option<Arc<ConfigHandle>>,
}

impl ApiConfig {
    #[must_use]
    pub fn new(manifest: PathBuf) -> Self {
        Self {
            manifests: vec![manifest],
            queue_timeout: Duration::from_secs(30),
            token: None,
            max_body_bytes: 32 * 1024 * 1024,
            max_loaded_models: 0,
            config: None,
        }
    }

    /// Build the runtime options from the durable application config.
    pub fn from_config(handle: Arc<ConfigHandle>) -> std::result::Result<Self, ConfigError> {
        let config = handle.snapshot()?;
        config.validate_for_serve()?;
        let base = handle.base_dir();
        let manifests = config
            .enabled_models()
            .map(|model| model.resolved_manifest(&base))
            .collect::<Vec<_>>();
        Ok(Self {
            manifests,
            queue_timeout: Duration::from_millis(config.server.queue_timeout_ms),
            token: config.server.token.clone(),
            max_body_bytes: config.server.max_body_bytes,
            max_loaded_models: config.server.max_loaded_models,
            config: Some(handle),
        })
    }

    #[must_use]
    pub fn with_config(mut self, handle: Arc<ConfigHandle>) -> Self {
        self.config = Some(handle);
        self
    }

    #[must_use]
    pub fn with_manifests(mut self, manifests: Vec<PathBuf>) -> Self {
        self.manifests = manifests;
        self
    }

    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}
