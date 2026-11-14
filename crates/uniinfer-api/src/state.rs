use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use uniinfer_core::{CancelToken, EngineError, ModelManifest, Result};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;
use uniinfer_registry::{EngineRegistry, SessionLease};

/// Server options.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub manifest: PathBuf,
    pub queue_timeout: Duration,
}

impl ApiConfig {
    #[must_use]
    pub fn new(manifest: PathBuf) -> Self {
        Self {
            manifest,
            queue_timeout: Duration::from_secs(30),
        }
    }
}

/// One loaded model, serialized by a single permit.
pub struct ModelSlot {
    pub manifest: ModelManifest,
    pub session: Arc<SessionLease>,
    pub permit: Arc<tokio::sync::Semaphore>,
}

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    pub(crate) slot: Arc<ModelSlot>,
    pub(crate) registry: Arc<EngineRegistry>,
    pub(crate) cancels: Arc<Mutex<HashMap<String, CancelToken>>>,
    pub(crate) active: Arc<AtomicUsize>,
    pub(crate) queue_timeout: Duration,
}

impl AppState {
    pub fn new(config: &ApiConfig) -> Result<Self> {
        let mut registry = EngineRegistry::new(DeviceInventory::probe());
        registry.register(Arc::new(OnnxEngine::new()?));

        let lease = registry.load(&config.manifest)?;
        let manifest = lease.manifest().clone();
        let tokenizer = manifest.tokenizer.as_deref().unwrap_or_default();
        if tokenizer != "bytes" {
            return Err(EngineError::UnsupportedEngine(format!(
                "tokenizer {tokenizer:?}; M1 serves byte tokenizers"
            )));
        }

        let slot = Arc::new(ModelSlot {
            manifest,
            session: Arc::new(lease),
            permit: Arc::new(tokio::sync::Semaphore::new(1)),
        });

        Ok(Self {
            slot,
            registry: Arc::new(registry),
            cancels: Arc::new(Mutex::new(HashMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
            queue_timeout: config.queue_timeout,
        })
    }

    /// Register a cancellation token for a request.
    pub(crate) fn register(&self, id: &str, token: CancelToken) -> Result<()> {
        self.cancels
            .lock()
            .map_err(|_| EngineError::Other("cancel table poisoned".into()))?
            .insert(id.to_owned(), token);
        Ok(())
    }

    pub(crate) fn unregister(&self, id: &str) {
        if let Ok(mut guard) = self.cancels.lock() {
            guard.remove(id);
        }
    }
}

/// Decrements the in-flight request gauge on drop.
pub(crate) struct ActiveGuard(Arc<AtomicUsize>);

impl ActiveGuard {
    pub(crate) fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}
