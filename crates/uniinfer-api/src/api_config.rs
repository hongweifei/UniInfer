use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_config::{ConfigError, ConfigHandle};

/// Server options.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub manifests: Vec<PathBuf>,
    pub model_roots: Vec<PathBuf>,
    pub queue_timeout: Duration,
    pub token: Option<String>,
    pub max_body_bytes: usize,
    pub max_loaded_models: usize,
    pub request_timeout: std::time::Duration,
    pub model_load_timeout: std::time::Duration,
    pub isolate_models: bool,
    pub vram_reserve_mb: u64,
    pub host_reserve_mb: u64,
    pub backend_dir: PathBuf,
    pub config: Option<Arc<ConfigHandle>>,
}

impl ApiConfig {
    #[must_use]
    pub fn new(manifest: PathBuf) -> Self {
        Self {
            model_roots: manifest
                .parent()
                .and_then(std::path::Path::parent)
                .map(|root| vec![root.to_path_buf()])
                .unwrap_or_default(),
            manifests: vec![manifest],
            queue_timeout: Duration::from_secs(30),
            token: None,
            max_body_bytes: 32 * 1024 * 1024,
            max_loaded_models: 0,
            request_timeout: std::time::Duration::ZERO,
            model_load_timeout: std::time::Duration::ZERO,
            isolate_models: false,
            vram_reserve_mb: 0,
            host_reserve_mb: 0,
            backend_dir: std::path::PathBuf::from("backends"),
            config: None,
        }
    }

    /// Build the runtime options from the durable application config.
    pub fn from_config(handle: Arc<ConfigHandle>) -> std::result::Result<Self, ConfigError> {
        let config = handle.snapshot()?;
        config.validate_for_serve()?;
        let backend_dir = config
            .server
            .backend_dir
            .clone()
            .map(|path| {
                if path.is_absolute() {
                    path
                } else {
                    handle.base_dir().join(path)
                }
            })
            .unwrap_or_else(|| handle.base_dir().join("backends"));
        Ok(Self {
            manifests: Vec::new(),
            model_roots: Vec::new(),
            queue_timeout: Duration::from_millis(config.server.queue_timeout_ms),
            token: config.server.token.clone(),
            max_body_bytes: config.server.max_body_bytes,
            max_loaded_models: config.server.max_loaded_models,
            request_timeout: std::time::Duration::from_millis(config.server.request_timeout_ms),
            model_load_timeout: std::time::Duration::from_millis(
                config.server.model_load_timeout_ms,
            ),
            isolate_models: config.server.isolate_models,
            vram_reserve_mb: config.server.vram_reserve_mb,
            host_reserve_mb: config.server.host_reserve_mb,
            backend_dir,
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
        if self.model_roots.is_empty() {
            self.model_roots = manifests
                .iter()
                .filter_map(|manifest| manifest.parent().and_then(std::path::Path::parent))
                .map(std::path::Path::to_path_buf)
                .collect();
            self.model_roots.sort();
            self.model_roots.dedup();
        }
        self.manifests = manifests;
        self
    }

    #[must_use]
    pub fn with_model_roots(mut self, model_roots: Vec<PathBuf>) -> Self {
        self.model_roots = model_roots;
        self
    }

    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}
