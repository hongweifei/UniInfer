use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::Mutex as AsyncMutex;
use uniinfer_backend_store::BackendStore;
use uniinfer_config::{ConfigHandle, ServerConfig};
use uniinfer_core::{CancelToken, CapabilityRequest, EngineError, ModelManifest, Result};
use uniinfer_model_inspect::{InspectOptions, inspect};
use uniinfer_service::{AudioSlot, ImageSlot, ModelSlot, Runtime};

use crate::api_config::ApiConfig;
use crate::cache::MediaCache;

/// A configured model that has not necessarily been loaded yet.
#[derive(Debug, Clone)]
pub(crate) struct CatalogEntry {
    pub(crate) path: PathBuf,
    pub(crate) manifest: ModelManifest,
    pub(crate) audio_only: bool,
    pub(crate) image_only: bool,
}

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    pub(crate) router: Arc<uniinfer_router::ModelRouter>,
    media: MediaCache,
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) token: Option<String>,
    pub(crate) max_body_bytes: usize,
    pub(crate) config: Option<Arc<ConfigHandle>>,
    pub(crate) manifest_paths: Arc<RwLock<HashMap<String, PathBuf>>>,
    pub(crate) model_roots: Vec<PathBuf>,
    request_timeout_ms: Arc<AtomicU64>,
    model_load_timeout_ms: Arc<AtomicU64>,
    pub(crate) catalog: Arc<RwLock<HashMap<String, CatalogEntry>>>,
    pub(crate) backends: Arc<BackendStore>,
    load_lock: Arc<AsyncMutex<()>>,
}

impl AppState {
    pub fn new(config: &ApiConfig) -> Result<Self> {
        let backends = Arc::new(BackendStore::new(config.backend_dir.clone()));
        let mut launches = Vec::new();
        match backends.installed() {
            Ok(installed) => {
                // Newest package first; duplicate family/variant variants are
                // resolved by the registry's declaration order.
                for backend in installed.iter().rev() {
                    match backend.launch() {
                        Ok(launch) => launches.push(launch),
                        Err(error) => {
                            tracing::warn!(%error, id = %backend.manifest().id, "skipping unusable backend package");
                        }
                    }
                }
            }
            Err(error) => {
                tracing::warn!(%error, "read backend store failed; using built-in engines");
            }
        }
        let runtime = Arc::new(Runtime::new_with_backends(
            config.isolate_models,
            &launches,
        )?);
        runtime.set_reserves(
            config.vram_reserve_mb.saturating_mul(1024 * 1024),
            config.host_reserve_mb.saturating_mul(1024 * 1024),
        );
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
            let audio_only = manifest.audio.is_some() && manifest.text.is_none();
            let image_only = manifest.image.is_some();
            manifest_paths.insert(id.clone(), path.clone());
            catalog.insert(
                id,
                CatalogEntry {
                    path: path.clone(),
                    manifest,
                    audio_only,
                    image_only,
                },
            );
        }

        Ok(Self {
            router: Arc::new(uniinfer_router::ModelRouter::new_with_limit(
                HashMap::new(),
                config.queue_timeout,
                config.max_loaded_models,
            )),
            media: MediaCache::new(config.max_loaded_models),
            runtime,
            token: config.token.clone(),
            max_body_bytes: config.max_body_bytes,
            config: config.config.clone(),
            manifest_paths: Arc::new(RwLock::new(manifest_paths)),
            model_roots: config.model_roots.clone(),
            request_timeout_ms: Arc::new(AtomicU64::new(
                u64::try_from(config.request_timeout.as_millis()).unwrap_or(u64::MAX),
            )),
            model_load_timeout_ms: Arc::new(AtomicU64::new(
                u64::try_from(config.model_load_timeout.as_millis()).unwrap_or(u64::MAX),
            )),
            catalog: Arc::new(RwLock::new(catalog)),
            backends,
            load_lock: Arc::new(AsyncMutex::new(())),
        })
    }

    /// Register sidecar packages installed after the runtime was built.
    pub(crate) fn sync_backends(&self) -> std::result::Result<(), String> {
        let installed = self
            .backends
            .installed()
            .map_err(|error| error.to_string())?;
        let mut launches = Vec::new();
        for backend in installed.iter().rev() {
            match backend.launch() {
                Ok(launch) => launches.push(launch),
                Err(error) => {
                    tracing::warn!(%error, id = %backend.manifest().id, "skipping unusable backend package");
                }
            }
        }
        self.runtime.register_sidecars(&launches);
        Ok(())
    }

    /// Select the first configured model that satisfies a capability request.
    /// Loaded models are preferred over lazy candidates.
    pub(crate) fn select_capability(
        &self,
        request: &CapabilityRequest,
        audio_only: bool,
        image_only: bool,
    ) -> Option<String> {
        let Ok(catalog) = self.catalog.read() else {
            return None;
        };
        let mut ids = catalog
            .iter()
            .filter(|(_, entry)| entry.audio_only == audio_only && entry.image_only == image_only)
            .filter(|(_, entry)| entry.manifest.supports(request))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        ids.sort();
        let loaded = ids.iter().find(|id| {
            if audio_only {
                self.media.contains_audio(id)
            } else if image_only {
                self.media.contains_image(id)
            } else {
                self.router.peek_slot(id).is_some()
            }
        });
        if let Some(id) = loaded {
            return Some(id.clone());
        }
        ids.into_iter().find(|id| {
            catalog
                .get(id)
                .is_some_and(|entry| self.runtime.probe_manifest(&entry.manifest).is_ok())
        })
    }

    pub(crate) fn catalog_entry(&self, id: &str) -> Option<CatalogEntry> {
        self.catalog.read().ok()?.get(id).cloned()
    }

    pub(crate) fn add_manifest(&self, path: PathBuf, manifest: ModelManifest) -> Result<()> {
        let id = manifest.id.clone();
        let audio_only = manifest.audio.is_some() && manifest.text.is_none();
        let image_only = manifest.image.is_some();
        self.manifest_paths
            .write()
            .map_err(|_| EngineError::Other("manifest paths lock poisoned".into()))?
            .insert(id.clone(), path.clone());
        self.catalog
            .write()
            .map_err(|_| EngineError::Other("catalog lock poisoned".into()))?
            .insert(
                id,
                CatalogEntry {
                    path,
                    manifest,
                    audio_only,
                    image_only,
                },
            );
        Ok(())
    }

    /// Rescan configured model roots and update the live catalog. This is the
    /// source of truth for downloaded/imported models; config stores no model
    /// list.
    pub(crate) fn refresh_catalog(&self) -> Result<()> {
        let mut found = Vec::new();
        for root in &self.model_roots {
            collect_manifests(root, 0, &mut found);
        }
        let mut catalog = self
            .catalog
            .write()
            .map_err(|_| EngineError::Other("catalog lock poisoned".into()))?;
        let mut paths = self
            .manifest_paths
            .write()
            .map_err(|_| EngineError::Other("manifest paths lock poisoned".into()))?;
        for path in found {
            let Ok(mut manifest) = ModelManifest::load(&path) else {
                continue;
            };
            if manifest.image.is_none() && looks_like_diffusion(&manifest) {
                let root = path.parent().unwrap_or_else(|| Path::new("."));
                if let Ok(inspection) = inspect(
                    root,
                    &InspectOptions {
                        id: Some(manifest.id.clone()),
                        ..InspectOptions::default()
                    },
                ) && inspection.manifest.image.is_some()
                {
                    if let Err(error) = inspection.write_manifest() {
                        tracing::warn!(%error, path = %path.display(), "repair diffusion manifest");
                    } else {
                        manifest = inspection.manifest;
                    }
                }
            }
            let id = manifest.id.clone();
            let audio_only = manifest.audio.is_some() && manifest.text.is_none();
            let image_only = manifest.image.is_some();
            paths.insert(id.clone(), path.clone());
            catalog.insert(
                id,
                CatalogEntry {
                    path,
                    manifest,
                    audio_only,
                    image_only,
                },
            );
        }
        let roots = self.model_roots.clone();
        catalog.retain(|id, entry| {
            if roots.iter().any(|root| entry.path.starts_with(root)) && !entry.path.is_file() {
                paths.remove(id);
                false
            } else {
                true
            }
        });
        Ok(())
    }

    pub(crate) fn request_timeout(&self) -> Option<std::time::Duration> {
        let ms = self.request_timeout_ms.load(Ordering::SeqCst);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    }

    fn model_load_timeout(&self) -> Option<std::time::Duration> {
        let ms = self.model_load_timeout_ms.load(Ordering::SeqCst);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    }

    pub(crate) fn apply_server_config(&self, server: &ServerConfig) {
        self.request_timeout_ms
            .store(server.request_timeout_ms, Ordering::SeqCst);
        self.model_load_timeout_ms
            .store(server.model_load_timeout_ms, Ordering::SeqCst);
        self.router
            .set_queue_timeout(std::time::Duration::from_millis(server.queue_timeout_ms));
        self.router.set_max_loaded(server.max_loaded_models);
        self.media.set_max_loaded(server.max_loaded_models);
    }

    pub(crate) fn has_model(&self, id: &str) -> bool {
        self.catalog
            .read()
            .is_ok_and(|catalog| catalog.contains_key(id))
    }

    /// Replace a model's manifest and unload its live slot so the next
    /// request loads with the new placement/context configuration.
    pub(crate) fn update_manifest(&self, id: &str, manifest: ModelManifest) -> Result<()> {
        let mut catalog = self
            .catalog
            .write()
            .map_err(|_| EngineError::Other("catalog lock poisoned".into()))?;
        let entry = catalog
            .get_mut(id)
            .ok_or_else(|| EngineError::ManifestInvalid(format!("model {id} is not configured")))?;
        entry.audio_only = manifest.audio.is_some() && manifest.text.is_none();
        entry.image_only = manifest.image.is_some();
        entry.manifest = manifest;
        drop(catalog);

        self.router.remove_slot(id);
        self.media.remove(id);
        Ok(())
    }

    pub(crate) async fn image_slot(&self, id: &str) -> Result<Arc<ImageSlot>> {
        if let Some(slot) = self.media.image(id) {
            return Ok(slot);
        }
        let entry = self
            .catalog_entry(id)
            .ok_or_else(|| EngineError::ManifestInvalid(format!("model {id} is not configured")))?;
        if !entry.image_only {
            return Err(EngineError::ManifestInvalid(format!(
                "model {id} has no [image] config"
            )));
        }
        let _guard = self.load_lock.lock().await;
        if let Some(slot) = self.media.image(id) {
            return Ok(slot);
        }
        let runtime = Arc::clone(&self.runtime);
        let manifest = entry.manifest;
        let base = entry.path.parent().map_or_else(
            || std::path::PathBuf::from("."),
            std::path::Path::to_path_buf,
        );
        let state = self;
        let slot = crate::loading::retry_memory(
            id,
            || {
                let runtime = Arc::clone(&runtime);
                let manifest = manifest.clone();
                let base = base.clone();
                async move {
                    tokio::task::spawn_blocking(move || {
                        runtime.load_image_manifest(manifest, &base)
                    })
                    .await
                    .map_err(|_| EngineError::Other("image load task panicked".into()))?
                }
            },
            |id| state.evict_for_memory(id),
        )
        .await?;
        self.media.insert_image(id, Arc::clone(&slot));
        Ok(slot)
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
        let timeout = self.model_load_timeout();
        let state = self;
        let slot = crate::loading::retry_memory(
            id,
            || {
                let runtime = Arc::clone(&runtime);
                let path = path.clone();
                let cancel = CancelToken::new();
                let task_cancel = cancel.clone();
                let task = tokio::task::spawn_blocking(move || {
                    runtime.load_cancellable(&path, task_cancel)
                });
                async move { crate::loading::await_load(timeout, task, cancel).await }
            },
            |id| state.evict_for_memory(id),
        )
        .await?;
        self.router.insert_slot(id, Arc::clone(&slot))?;
        Ok(slot)
    }

    pub(crate) async fn audio_slot(&self, id: &str) -> Result<Arc<AudioSlot>> {
        if let Some(slot) = self.media.audio(id) {
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
        if let Some(slot) = self.media.audio(id) {
            return Ok(slot);
        }
        let runtime = Arc::clone(&self.runtime);
        let path = entry.path;
        let timeout = self.model_load_timeout();
        let state = self;
        let slot = crate::loading::retry_memory(
            id,
            || {
                let runtime = Arc::clone(&runtime);
                let path = path.clone();
                let cancel = CancelToken::new();
                let task_cancel = cancel.clone();
                let task = tokio::task::spawn_blocking(move || {
                    runtime.load_audio_cancellable(&path, task_cancel)
                });
                async move { crate::loading::await_load(timeout, task, cancel).await }
            },
            |id| state.evict_for_memory(id),
        )
        .await?;
        self.media.insert_audio(id, Arc::clone(&slot));
        Ok(slot)
    }

    pub(crate) fn unload_text(&self, id: &str) -> bool {
        self.router.remove_slot(id)
    }

    pub(crate) fn cached_audio(&self, id: &str) -> Option<Arc<AudioSlot>> {
        self.media.audio(id)
    }

    pub(crate) fn cached_image(&self, id: &str) -> Option<Arc<ImageSlot>> {
        self.media.image(id)
    }

    pub(crate) fn media_counts(&self) -> (usize, usize) {
        self.media.counts()
    }

    pub(crate) fn unload_audio(&self, id: &str) -> bool {
        self.media.remove_audio(id)
    }

    pub(crate) fn unload_image(&self, id: &str) -> bool {
        self.media.remove_image(id)
    }

    /// Evict one cached model to make room, never evicting the model being loaded.
    pub(crate) fn evict_for_memory(&self, except: &str) -> bool {
        if self.router.evict_lru_except(Some(except)).is_some() {
            self.media.record_eviction();
            return true;
        }
        self.media.evict_one(except)
    }

    pub(crate) fn memory_evictions(&self) -> u64 {
        self.media.evictions()
    }

    pub(crate) fn audio_catalog_ids(&self) -> Vec<String> {
        self.catalog
            .read()
            .map(|catalog| {
                catalog
                    .iter()
                    .filter(|(_, entry)| entry.audio_only)
                    .map(|(id, _)| id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn looks_like_diffusion(manifest: &ModelManifest) -> bool {
    let has = |prefix: &str| {
        manifest
            .files
            .iter()
            .any(|file| file.path.to_string_lossy().starts_with(prefix))
    };
    has("text_encoder/") && has("unet/") && has("vae_decoder/")
}

fn collect_manifests(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_manifests(&path, depth + 1, out);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("manifest") && name.ends_with(".toml"))
        {
            out.push(path);
        }
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

    #[test]
    fn refresh_catalog_discovers_and_removes_model_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models_root = temp.path().join("models");
        std::fs::create_dir_all(&models_root).expect("models root");
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
        let config = ApiConfig::new(fixture.clone()).with_model_roots(vec![models_root.clone()]);
        let state = AppState::new(&config).expect("state");
        assert!(state.catalog_entry("discovered").is_none());

        let model_dir = models_root.join("discovered");
        std::fs::create_dir_all(&model_dir).expect("model dir");
        let mut manifest = ModelManifest::load(&fixture).expect("manifest");
        manifest.id = "discovered".into();
        manifest
            .save(&model_dir.join("manifest.toml"))
            .expect("save");
        state.refresh_catalog().expect("refresh");
        assert!(state.catalog_entry("discovered").is_some());

        std::fs::remove_dir_all(&model_dir).expect("remove");
        state.refresh_catalog().expect("refresh");
        assert!(state.catalog_entry("discovered").is_none());
    }

    #[test]
    fn refresh_repairs_legacy_diffusion_manifest() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models_root = temp.path().join("models");
        let model_dir = models_root.join("legacy-diffusion");
        for (source, target) in [
            ("text_encoder.onnx", "text_encoder/model.onnx"),
            ("unet.onnx", "unet/model.onnx"),
            ("vae_decoder.onnx", "vae_decoder/model.onnx"),
        ] {
            let target = model_dir.join(target);
            std::fs::create_dir_all(target.parent().unwrap()).expect("component dir");
            std::fs::copy(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../models/tiny-diffusion")
                    .join(source),
                target,
            )
            .expect("copy component");
        }
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("tokenizer");
        std::fs::create_dir_all(model_dir.join("text_encoder")).expect("config dir");
        std::fs::write(
            model_dir.join("text_encoder/config.json"),
            r#"{"architectures":["CLIPTextModel"],"vocab_size":100}"#,
        )
        .expect("config");

        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/tiny-diffusion/manifest.toml");
        let mut manifest = ModelManifest::load(&fixture).expect("manifest");
        manifest.id = "legacy-diffusion".into();
        manifest.implementation = "onnx.graph".into();
        manifest.image = None;
        manifest.files = vec![
            uniinfer_core::ModelFile {
                path: "text_encoder/model.onnx".into(),
                sha256: "00".repeat(32),
            },
            uniinfer_core::ModelFile {
                path: "unet/model.onnx".into(),
                sha256: "00".repeat(32),
            },
            uniinfer_core::ModelFile {
                path: "vae_decoder/model.onnx".into(),
                sha256: "00".repeat(32),
            },
        ];
        manifest.capabilities = vec![uniinfer_core::Capability {
            name: "text_chat".into(),
            operation: uniinfer_core::Operation::Generate,
            required_inputs: vec![uniinfer_core::Modality::Text],
            optional_inputs: Vec::new(),
            outputs: vec![uniinfer_core::Modality::Text],
            mode: uniinfer_core::Mode::Stream,
        }];
        manifest
            .save(&model_dir.join("manifest.toml"))
            .expect("save legacy manifest");

        let config = ApiConfig::new(fixture).with_model_roots(vec![models_root]);
        let state = AppState::new(&config).expect("state");
        state.refresh_catalog().expect("refresh");
        let entry = state.catalog_entry("legacy-diffusion").expect("repaired");
        assert_eq!(entry.manifest.implementation, "onnx.diffusion");
        assert!(entry.manifest.image.is_some());
        assert_eq!(entry.manifest.capabilities[0].name, "text_to_image");
    }

    #[test]
    fn update_manifest_refreshes_catalog_without_restart() {
        let state = state(std::time::Duration::ZERO);
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
        let mut manifest = ModelManifest::load(&path).expect("manifest");
        manifest.context_len = 123;
        state
            .update_manifest("tiny-byte-onnx", manifest)
            .expect("update");
        let entry = state.catalog_entry("tiny-byte-onnx").expect("catalog");
        assert_eq!(entry.manifest.context_len, 123);
    }

    #[tokio::test]
    async fn load_past_deadline_returns_timeout_and_cancels_task() {
        let state = state(std::time::Duration::from_millis(5));
        let cancel = CancelToken::new();
        let error =
            crate::loading::await_load(state.model_load_timeout(), slow_load(), cancel.clone())
                .await
                .expect_err("timeout");
        assert!(matches!(error, EngineError::Timeout));
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn load_finishing_after_budget_still_times_out() {
        let state = state(std::time::Duration::from_millis(1));
        let cancel = CancelToken::new();
        let task = slow_load();
        let error = crate::loading::await_load(state.model_load_timeout(), task, cancel.clone())
            .await
            .expect_err("timeout");
        assert!(matches!(error, EngineError::Timeout));
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn load_within_deadline_returns_value() {
        let state = state(std::time::Duration::from_secs(5));
        let cancel = CancelToken::new();
        crate::loading::await_load(state.model_load_timeout(), slow_load(), cancel.clone())
            .await
            .expect("load");
        assert!(!cancel.is_cancelled());
    }

    #[tokio::test]
    async fn zero_deadline_never_times_out() {
        let state = state(std::time::Duration::ZERO);
        let cancel = CancelToken::new();
        crate::loading::await_load(state.model_load_timeout(), slow_load(), cancel.clone())
            .await
            .expect("load");
        assert!(!cancel.is_cancelled());
    }
}
