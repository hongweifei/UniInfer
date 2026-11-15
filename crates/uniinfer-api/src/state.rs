use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tokio::sync::Mutex as AsyncMutex;
use uniinfer_config::ConfigHandle;
use uniinfer_core::{CancelToken, EngineError, Modality, ModelManifest, Result};
use uniinfer_service::{AudioSlot, ModelSlot, Runtime};

use crate::api_config::ApiConfig;

/// A configured model that has not necessarily been loaded yet.
#[derive(Debug, Clone)]
pub(crate) struct CatalogEntry {
    pub(crate) path: PathBuf,
    pub(crate) manifest: ModelManifest,
    pub(crate) audio_only: bool,
}

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    pub(crate) router: Arc<uniinfer_router::ModelRouter>,
    pub(crate) audio_slots: Arc<Mutex<HashMap<String, Arc<AudioSlot>>>>,
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) token: Option<String>,
    pub(crate) max_body_bytes: usize,
    pub(crate) config: Option<Arc<ConfigHandle>>,
    pub(crate) manifest_paths: Arc<HashMap<String, PathBuf>>,
    pub(crate) request_timeout: Option<std::time::Duration>,
    model_load_timeout: Option<std::time::Duration>,
    pub(crate) catalog: Arc<HashMap<String, CatalogEntry>>,
    load_lock: Arc<AsyncMutex<()>>,
    audio_lru: Arc<Mutex<VecDeque<String>>>,
    max_loaded_models: usize,
}

impl AppState {
    pub fn new(config: &ApiConfig) -> Result<Self> {
        if config.manifests.is_empty() {
            return Err(EngineError::ManifestInvalid(
                "no manifests configured".into(),
            ));
        }
        let runtime = Arc::new(Runtime::new()?);
        let mut manifest_paths = HashMap::new();
        let mut catalog = HashMap::new();
        for path in &config.manifests {
            let manifest = ModelManifest::load(path)?;
            let id = manifest.id.clone();
            if catalog.contains_key(&id) {
                return Err(EngineError::ManifestInvalid(format!(
                    "duplicate model id {id}"
                )));
            }
            let audio_only =
                manifest.modalities.contains(&Modality::Audio) && manifest.text.is_none();
            manifest_paths.insert(id.clone(), path.clone());
            catalog.insert(
                id,
                CatalogEntry {
                    path: path.clone(),
                    manifest,
                    audio_only,
                },
            );
        }

        Ok(Self {
            router: Arc::new(uniinfer_router::ModelRouter::new_with_limit(
                HashMap::new(),
                config.queue_timeout,
                config.max_loaded_models,
            )),
            audio_slots: Arc::new(Mutex::new(HashMap::new())),
            runtime,
            token: config.token.clone(),
            max_body_bytes: config.max_body_bytes,
            config: config.config.clone(),
            manifest_paths: Arc::new(manifest_paths),
            request_timeout: (config.request_timeout > std::time::Duration::ZERO)
                .then_some(config.request_timeout),
            model_load_timeout: (config.model_load_timeout > std::time::Duration::ZERO)
                .then_some(config.model_load_timeout),
            catalog: Arc::new(catalog),
            load_lock: Arc::new(AsyncMutex::new(())),
            audio_lru: Arc::new(Mutex::new(VecDeque::new())),
            max_loaded_models: config.max_loaded_models,
        })
    }

    pub(crate) fn catalog_entry(&self, id: &str) -> Option<CatalogEntry> {
        self.catalog.get(id).cloned()
    }

    pub(crate) async fn text_slot(&self, id: &str) -> Result<Arc<ModelSlot>> {
        if let Some(slot) = self.router.slot(id) {
            return Ok(slot);
        }
        let entry = self
            .catalog_entry(id)
            .ok_or_else(|| EngineError::ManifestInvalid(format!("model {id} is not configured")))?;
        if entry.audio_only {
            return Err(EngineError::ManifestInvalid(format!(
                "model {id} is audio-only"
            )));
        }
        let _guard = self.load_lock.lock().await;
        if let Some(slot) = self.router.slot(id) {
            return Ok(slot);
        }
        let runtime = Arc::clone(&self.runtime);
        let path = entry.path;
        let cancel = CancelToken::new();
        let task_cancel = cancel.clone();
        let task =
            tokio::task::spawn_blocking(move || runtime.load_cancellable(&path, task_cancel));
        let slot = self.await_load(task, cancel).await?;
        self.router.insert_slot(id, Arc::clone(&slot))?;
        Ok(slot)
    }

    pub(crate) async fn audio_slot(&self, id: &str) -> Result<Arc<AudioSlot>> {
        if let Some(slot) = self
            .audio_slots
            .lock()
            .ok()
            .and_then(|slots| slots.get(id).cloned())
        {
            return Ok(slot);
        }
        let entry = self
            .catalog_entry(id)
            .ok_or_else(|| EngineError::ManifestInvalid(format!("model {id} is not configured")))?;
        if !entry.audio_only {
            return Err(EngineError::ManifestInvalid(format!(
                "model {id} is not an audio-only model"
            )));
        }
        let _guard = self.load_lock.lock().await;
        if let Some(slot) = self
            .audio_slots
            .lock()
            .ok()
            .and_then(|slots| slots.get(id).cloned())
        {
            return Ok(slot);
        }
        let runtime = Arc::clone(&self.runtime);
        let path = entry.path;
        let cancel = CancelToken::new();
        let task_cancel = cancel.clone();
        let task =
            tokio::task::spawn_blocking(move || runtime.load_audio_cancellable(&path, task_cancel));
        let slot = self.await_load(task, cancel).await?;
        self.audio_slots
            .lock()
            .map_err(|_| EngineError::Other("audio catalog lock poisoned".into()))?
            .insert(id.to_owned(), Arc::clone(&slot));
        self.touch_audio(id);
        self.evict_audio();
        Ok(slot)
    }

    fn touch_audio(&self, id: &str) {
        if let Ok(mut lru) = self.audio_lru.lock() {
            lru.retain(|entry| entry != id);
            lru.push_back(id.to_owned());
        }
    }

    fn evict_audio(&self) {
        if self.max_loaded_models == 0 {
            return;
        }
        let Ok(mut lru) = self.audio_lru.lock() else {
            return;
        };
        while lru.len() > self.max_loaded_models {
            let Some(oldest) = lru.pop_front() else {
                break;
            };
            if let Ok(mut slots) = self.audio_slots.lock() {
                slots.remove(&oldest);
            }
        }
    }

    async fn await_load<T>(
        &self,
        task: tokio::task::JoinHandle<Result<T>>,
        cancel: CancelToken,
    ) -> Result<T> {
        let joined = match self.model_load_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, task).await {
                Ok(joined) => joined,
                Err(_) => {
                    cancel.cancel();
                    return Err(EngineError::Timeout);
                }
            },
            None => task.await,
        };
        joined.map_err(|error| EngineError::Other(format!("model load task: {error}")))?
    }

    pub(crate) fn unload_text(&self, id: &str) -> bool {
        self.router.remove_slot(id)
    }

    pub(crate) fn unload_audio(&self, id: &str) -> bool {
        let removed = self
            .audio_slots
            .lock()
            .map(|mut slots| slots.remove(id).is_some())
            .unwrap_or(false);
        if let Ok(mut lru) = self.audio_lru.lock() {
            lru.retain(|entry| entry != id);
        }
        removed
    }

    pub(crate) fn audio_catalog_ids(&self) -> Vec<String> {
        self.catalog
            .iter()
            .filter(|(_, entry)| entry.audio_only)
            .map(|(id, _)| id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(model_load_timeout: std::time::Duration) -> AppState {
        let manifest =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
        let mut config = ApiConfig::new(manifest);
        config.model_load_timeout = model_load_timeout;
        AppState::new(&config).expect("state")
    }

    fn slow_load() -> tokio::task::JoinHandle<Result<()>> {
        tokio::task::spawn_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(200));
            Ok(())
        })
    }

    #[tokio::test]
    async fn load_past_deadline_returns_timeout_and_cancels_task() {
        let state = state(std::time::Duration::from_millis(5));
        let cancel = CancelToken::new();
        let error = state
            .await_load(slow_load(), cancel.clone())
            .await
            .expect_err("timeout");
        assert!(matches!(error, EngineError::Timeout));
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn load_within_deadline_returns_value() {
        let state = state(std::time::Duration::from_secs(5));
        let cancel = CancelToken::new();
        state
            .await_load(slow_load(), cancel.clone())
            .await
            .expect("load");
        assert!(!cancel.is_cancelled());
    }

    #[tokio::test]
    async fn zero_deadline_never_times_out() {
        let state = state(std::time::Duration::ZERO);
        let cancel = CancelToken::new();
        state
            .await_load(slow_load(), cancel.clone())
            .await
            .expect("load");
        assert!(!cancel.is_cancelled());
    }
}
