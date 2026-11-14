use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use uniinfer_core::{CancelToken, EngineError, Result};
use uniinfer_service::{ModelSlot, Runtime};

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
    pub(crate) slots: Arc<HashMap<String, Arc<ModelSlot>>>,
    pub(crate) runtime: Arc<Runtime>,
    pub(crate) cancels: Arc<Mutex<HashMap<String, CancelToken>>>,
    pub(crate) active: Arc<AtomicUsize>,
    pub(crate) queue_timeout: Duration,
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
            slots: Arc::new(slots),
            runtime,
            cancels: Arc::new(Mutex::new(HashMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
            queue_timeout: config.queue_timeout,
            token: config.token.clone(),
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

/// Cancels and unregisters a request when the handler is dropped.
pub(crate) struct CancelGuard {
    cancels: Arc<Mutex<HashMap<String, CancelToken>>>,
    id: String,
    token: CancelToken,
    armed: bool,
}

impl CancelGuard {
    pub(crate) fn new(state: &AppState, id: String, token: CancelToken) -> Result<Self> {
        state.register(&id, token.clone())?;
        Ok(Self {
            cancels: Arc::clone(&state.cancels),
            id,
            token,
            armed: true,
        })
    }

    pub(crate) fn token(&self) -> CancelToken {
        self.token.clone()
    }

    /// Hand cleanup to the generation task (streaming returns immediately).
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.token.cancel();
        if let Ok(mut guard) = self.cancels.lock() {
            guard.remove(&self.id);
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn state() -> AppState {
        let manifest =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
        AppState::new(&ApiConfig::new(manifest)).expect("state")
    }

    #[test]
    fn guard_cancels_and_unregisters_on_drop() {
        let state = state();
        let token = CancelToken::new();
        {
            let _guard =
                CancelGuard::new(&state, "sess-test".into(), token.clone()).expect("guard");
        }
        assert!(token.is_cancelled());
        assert!(
            !state
                .cancels
                .lock()
                .expect("lock")
                .contains_key("sess-test")
        );
    }

    #[test]
    fn disarmed_guard_leaves_cleanup_to_task() {
        let state = state();
        let token = CancelToken::new();
        {
            let mut guard =
                CancelGuard::new(&state, "sess-test".into(), token.clone()).expect("guard");
            guard.disarm();
        }
        assert!(!token.is_cancelled());
        assert!(
            state
                .cancels
                .lock()
                .expect("lock")
                .contains_key("sess-test")
        );
        state.unregister("sess-test");
    }
}
