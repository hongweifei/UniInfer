//! Model catalog and session lifecycle shared by HTTP, WebSocket and CLI.
//!
//! The router owns the loaded model slots, the cancellation table and the
//! active-session gauge. Transport crates keep DTO parsing and wire formats.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use uniinfer_core::{CancelToken, EngineError, Result};
use uniinfer_service::ModelSlot;

/// Loaded models plus their request lifecycle state.
pub struct ModelRouter {
    slots: Arc<HashMap<String, Arc<ModelSlot>>>,
    cancels: Arc<Mutex<HashMap<String, CancelToken>>>,
    active: Arc<AtomicUsize>,
    queue_timeout: Duration,
}

impl ModelRouter {
    #[must_use]
    pub fn new(slots: HashMap<String, Arc<ModelSlot>>, queue_timeout: Duration) -> Self {
        Self {
            slots: Arc::new(slots),
            cancels: Arc::new(Mutex::new(HashMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
            queue_timeout,
        }
    }

    #[must_use]
    pub fn slots(&self) -> &HashMap<String, Arc<ModelSlot>> {
        &self.slots
    }

    #[must_use]
    pub fn slot(&self, id: &str) -> Option<Arc<ModelSlot>> {
        self.slots.get(id).cloned()
    }

    #[must_use]
    pub fn queue_timeout(&self) -> Duration {
        self.queue_timeout
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    /// Register a request-scoped cancellation token.
    pub fn register(&self, id: &str, token: CancelToken) -> Result<CancelGuard> {
        self.cancels
            .lock()
            .map_err(|_| EngineError::Other("cancel table poisoned".into()))?
            .insert(id.to_owned(), token.clone());
        Ok(CancelGuard {
            cancels: Arc::clone(&self.cancels),
            id: id.to_owned(),
            token,
            armed: true,
        })
    }

    pub fn unregister(&self, id: &str) {
        if let Ok(mut guard) = self.cancels.lock() {
            guard.remove(id);
        }
    }

    /// Cancel a registered request. Returns false when the id is unknown.
    #[must_use]
    pub fn cancel(&self, id: &str) -> bool {
        let token = self
            .cancels
            .lock()
            .ok()
            .and_then(|guard| guard.get(id).cloned());
        if let Some(token) = token {
            token.cancel();
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn active_guard(&self) -> ActiveGuard {
        ActiveGuard::new(Arc::clone(&self.active))
    }
}

/// Cancels and unregisters a request when the handler is dropped.
pub struct CancelGuard {
    cancels: Arc<Mutex<HashMap<String, CancelToken>>>,
    id: String,
    token: CancelToken,
    armed: bool,
}

impl CancelGuard {
    #[must_use]
    pub fn token(&self) -> CancelToken {
        self.token.clone()
    }

    /// Hand cleanup to the generation task (streaming returns immediately).
    pub fn disarm(&mut self) {
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
pub struct ActiveGuard(Arc<AtomicUsize>);

impl ActiveGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use super::*;

    fn router() -> ModelRouter {
        ModelRouter::new(HashMap::new(), Duration::from_millis(10))
    }

    #[test]
    fn guard_cancels_and_unregisters_on_drop() {
        let router = router();
        let token = CancelToken::new();
        {
            let _guard = router.register("sess-test", token.clone()).expect("guard");
        }
        assert!(token.is_cancelled());
        assert!(!router.cancel("sess-test"));
    }

    #[test]
    fn cancel_marks_registered_token() {
        let router = router();
        let token = CancelToken::new();
        let _guard = router.register("sess-test", token.clone()).expect("guard");
        assert!(router.cancel("sess-test"));
        assert!(token.is_cancelled());
        assert!(!router.cancel("missing"));
    }

    #[test]
    fn active_guard_tracks_in_flight_requests() {
        let router = router();
        let guard = router.active_guard();
        assert_eq!(router.active_count(), 1);
        drop(guard);
        assert_eq!(router.active_count(), 0);
    }
}
