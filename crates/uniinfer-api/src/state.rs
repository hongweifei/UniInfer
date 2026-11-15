use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_core::{EngineError, Result};
use uniinfer_service::Runtime;

/// Server options.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub manifests: Vec<PathBuf>,
    pub queue_timeout: Duration,
    pub token: Option<String>,
}

impl ApiConfig {
    #[must_use]
    pub fn new(manifest: PathBuf) -> Self {
        Self {
            manifests: vec![manifest],
            queue_timeout: Duration::from_secs(30),
            token: None,
        }
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

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    pub(crate) router: Arc<uniinfer_router::ModelRouter>,
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) token: Option<String>,
}

impl AppState {
    pub fn new(config: &ApiConfig) -> Result<Self> {
        if config.manifests.is_empty() {
            return Err(EngineError::ManifestInvalid(
                "no manifests configured".into(),
            ));
        }
        let runtime = Arc::new(Runtime::new()?);
        let mut slots = HashMap::new();
        for manifest in &config.manifests {
            let slot = runtime.load(manifest)?;
            let id = slot.manifest().id.clone();
            if slots.insert(id.clone(), slot).is_some() {
                return Err(EngineError::ManifestInvalid(format!(
                    "duplicate model id {id}"
                )));
            }
        }

        Ok(Self {
            router: Arc::new(uniinfer_router::ModelRouter::new(
                slots,
                config.queue_timeout,
            )),
            runtime,
            token: config.token.clone(),
        })
    }
}
