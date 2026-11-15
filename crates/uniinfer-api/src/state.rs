use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_config::{ConfigError, ConfigHandle};
use uniinfer_core::{EngineError, Modality, ModelManifest, Result};
use uniinfer_service::AudioSlot;
use uniinfer_service::Runtime;

/// Server options.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub manifests: Vec<PathBuf>,
    pub queue_timeout: Duration,
    pub token: Option<String>,
    pub max_body_bytes: usize,
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

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    pub(crate) router: Arc<uniinfer_router::ModelRouter>,
    pub(crate) audio_slots: Arc<HashMap<String, Arc<AudioSlot>>>,
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) token: Option<String>,
    pub(crate) max_body_bytes: usize,
    pub(crate) config: Option<Arc<ConfigHandle>>,
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
        let mut audio_slots = HashMap::new();
        for path in &config.manifests {
            let manifest = ModelManifest::load(path)?;
            let audio_only =
                manifest.modalities.contains(&Modality::Audio) && manifest.text.is_none();
            let id = manifest.id.clone();
            if slots.contains_key(&id) || audio_slots.contains_key(&id) {
                return Err(EngineError::ManifestInvalid(format!(
                    "duplicate model id {id}"
                )));
            }
            if audio_only {
                let slot = runtime.load_audio(path)?;
                if audio_slots.insert(id.clone(), slot).is_some() {
                    return Err(EngineError::ManifestInvalid(format!(
                        "duplicate model id {id}"
                    )));
                }
            } else {
                let slot = runtime.load(path)?;
                if slots.insert(id.clone(), slot).is_some() {
                    return Err(EngineError::ManifestInvalid(format!(
                        "duplicate model id {id}"
                    )));
                }
            }
        }

        Ok(Self {
            router: Arc::new(uniinfer_router::ModelRouter::new(
                slots,
                config.queue_timeout,
            )),
            audio_slots: Arc::new(audio_slots),
            runtime,
            token: config.token.clone(),
            max_body_bytes: config.max_body_bytes,
            config: config.config.clone(),
        })
    }
}
