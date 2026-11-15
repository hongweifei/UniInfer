use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_core::{EngineError, Modality, ModelManifest, Result};
#[cfg(feature = "whisper")]
use uniinfer_service::AudioSlot;
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
    #[cfg(feature = "whisper")]
    pub(crate) audio_slots: Arc<HashMap<String, Arc<AudioSlot>>>,
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
        #[cfg(feature = "whisper")]
        let mut audio_slots = HashMap::new();
        for path in &config.manifests {
            let manifest = ModelManifest::load(path)?;
            let audio_only =
                manifest.modalities.contains(&Modality::Audio) && manifest.text.is_none();
            let id = manifest.id.clone();
            #[cfg(feature = "whisper")]
            if slots.contains_key(&id) || audio_slots.contains_key(&id) {
                return Err(EngineError::ManifestInvalid(format!(
                    "duplicate model id {id}"
                )));
            }
            #[cfg(not(feature = "whisper"))]
            if slots.contains_key(&id) {
                return Err(EngineError::ManifestInvalid(format!(
                    "duplicate model id {id}"
                )));
            }
            if audio_only {
                #[cfg(feature = "whisper")]
                {
                    let slot = runtime.load_audio(path)?;
                    if audio_slots.insert(id.clone(), slot).is_some() {
                        return Err(EngineError::ManifestInvalid(format!(
                            "duplicate model id {id}"
                        )));
                    }
                }
                #[cfg(not(feature = "whisper"))]
                {
                    return Err(EngineError::UnsupportedModality {
                        format: manifest.format,
                        modality: Modality::Audio,
                    });
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
            #[cfg(feature = "whisper")]
            audio_slots: Arc::new(audio_slots),
            runtime,
            token: config.token.clone(),
        })
    }
}
