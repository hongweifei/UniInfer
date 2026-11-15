use std::collections::HashMap;
use std::sync::Arc;

use uniinfer_config::ConfigHandle;
use uniinfer_core::{EngineError, Modality, ModelManifest, Result};
use uniinfer_service::AudioSlot;
use uniinfer_service::Runtime;

use crate::api_config::ApiConfig;

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
