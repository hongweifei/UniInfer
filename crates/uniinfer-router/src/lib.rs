//! Model catalog and session lifecycle shared by HTTP, WebSocket and CLI.
//!
//! The router owns the loaded model slots, the cancellation table and the
//! active-session gauge. Transport crates keep DTO parsing and wire formats.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use thiserror::Error;
use uniinfer_core::{CancelToken, EngineError, Result};
use uniinfer_service::ModelSlot;

mod metrics;
pub use metrics::{MetricsSnapshot, ModelMetrics};

/// Errors raised while preparing a text request.
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("model {0} is not loaded")]
    ModelNotFound(String),
    #[error("prompt uses {tokens} tokens, context_len is {limit}")]
    ContextOverflow { tokens: usize, limit: usize },
    #[error("max_tokens resolves to zero")]
    InvalidMaxTokens,
    #[error("model is busy")]
    Busy,
    #[error(transparent)]
    Engine(#[from] EngineError),
}

impl RouterError {
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::ModelNotFound(_) => "model_not_found",
            Self::ContextOverflow { .. } => "context_overflow",
            Self::InvalidMaxTokens => "invalid_max_tokens",
            Self::Busy => "busy",
            Self::Engine(error) => error.code(),
        }
    }
}

/// A prepared text request holding its lease, permit and cancellation guard.
pub struct PreparedText {
    pub slot: Arc<ModelSlot>,
    pub permit: tokio::sync::OwnedSemaphorePermit,
    pub session_id: String,
    pub prompt: String,
    pub images: Vec<Vec<u8>>,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    pub cancel: CancelToken,
    pub guard: CancelGuard,
    pub started: Instant,
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

/// Loaded models plus their request lifecycle state.
type CancelTable = Arc<Mutex<HashMap<String, CancelEntry>>>;

#[derive(Clone)]
struct CancelEntry {
    token: CancelToken,
    model: Option<String>,
}

pub struct ModelRouter {
    slots: Arc<RwLock<HashMap<String, Arc<ModelSlot>>>>,
    cancels: CancelTable,
    active: Arc<AtomicUsize>,
    metrics: metrics::Metrics,
    queue_timeout_ms: Arc<AtomicU64>,
    max_loaded: Arc<AtomicUsize>,
    lru: Arc<Mutex<VecDeque<String>>>,
}

impl ModelRouter {
    #[must_use]
    pub fn new(slots: HashMap<String, Arc<ModelSlot>>, queue_timeout: Duration) -> Self {
        Self::new_with_limit(slots, queue_timeout, 0)
    }

    #[must_use]
    pub fn new_with_limit(
        slots: HashMap<String, Arc<ModelSlot>>,
        queue_timeout: Duration,
        max_loaded: usize,
    ) -> Self {
        let lru = slots.keys().cloned().collect::<VecDeque<_>>();
        Self {
            slots: Arc::new(RwLock::new(slots)),
            cancels: Arc::new(Mutex::new(HashMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
            metrics: metrics::Metrics::default(),
            queue_timeout_ms: Arc::new(AtomicU64::new(
                u64::try_from(queue_timeout.as_millis()).unwrap_or(u64::MAX),
            )),
            max_loaded: Arc::new(AtomicUsize::new(max_loaded)),
            lru: Arc::new(Mutex::new(lru)),
        }
    }

    #[must_use]
    pub fn slots(&self) -> Vec<Arc<ModelSlot>> {
        self.slots
            .read()
            .map(|slots| slots.values().cloned().collect())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn slot(&self, id: &str) -> Option<Arc<ModelSlot>> {
        let slot = self
            .slots
            .read()
            .ok()
            .and_then(|slots| slots.get(id).cloned())?;
        self.touch(id);
        Some(slot)
    }

    /// Read a loaded slot without changing its LRU position.
    #[must_use]
    pub fn peek_slot(&self, id: &str) -> Option<Arc<ModelSlot>> {
        self.slots
            .read()
            .ok()
            .and_then(|slots| slots.get(id).cloned())
    }

    /// Evict the least-recently-used cached model other than `except`.
    /// The returned slot may still be alive through an in-flight request.
    pub fn evict_lru_except(&self, except: Option<&str>) -> Option<String> {
        let mut lru = self.lru.lock().ok()?;
        loop {
            let id = lru.pop_front()?;
            if except.is_some_and(|except| except == id) {
                continue;
            }
            let removed = self
                .slots
                .write()
                .map(|mut slots| slots.remove(&id).is_some())
                .unwrap_or(false);
            if removed {
                return Some(id);
            }
        }
    }

    /// Drop a loaded slot from the cache. Existing request `Arc`s keep it alive.
    pub fn remove_slot(&self, id: &str) -> bool {
        let removed = self
            .slots
            .write()
            .map(|mut slots| slots.remove(id).is_some())
            .unwrap_or(false);
        if let Ok(mut lru) = self.lru.lock() {
            lru.retain(|entry| entry != id);
        }
        removed
    }

    /// Insert a lazily loaded model. Replacing an existing entry is a no-op
    /// when another request won the load race.
    pub fn insert_slot(&self, id: &str, slot: Arc<ModelSlot>) -> Result<()> {
        {
            let mut slots = self
                .slots
                .write()
                .map_err(|_| EngineError::Other("model catalog lock poisoned".into()))?;
            slots.entry(id.to_owned()).or_insert(slot);
        }
        self.touch(id);
        self.evict();
        Ok(())
    }

    fn touch(&self, id: &str) {
        if let Ok(mut lru) = self.lru.lock() {
            lru.retain(|entry| entry != id);
            lru.push_back(id.to_owned());
        }
    }

    fn evict(&self) {
        let max_loaded = self.max_loaded.load(Ordering::SeqCst);
        if max_loaded == 0 {
            return;
        }
        let Ok(mut lru) = self.lru.lock() else {
            return;
        };
        while lru.len() > max_loaded {
            let Some(oldest) = lru.pop_front() else {
                break;
            };
            if let Ok(mut slots) = self.slots.write() {
                slots.remove(&oldest);
            }
        }
    }

    /// Resolve a model, validate context, acquire its permit and register
    /// cancellation. Waits up to the router's queue timeout.
    pub async fn prepare_text(
        &self,
        model: &str,
        prompt: String,
        requested: Option<usize>,
    ) -> std::result::Result<PreparedText, RouterError> {
        self.prepare_text_with(model, prompt, requested, Some(self.queue_timeout()))
            .await
    }

    /// Non-blocking variant used by realtime sessions.
    pub async fn prepare_text_now(
        &self,
        model: &str,
        prompt: String,
        requested: Option<usize>,
    ) -> std::result::Result<PreparedText, RouterError> {
        self.prepare_text_with(model, prompt, requested, None).await
    }

    async fn prepare_text_with(
        &self,
        model: &str,
        prompt: String,
        requested: Option<usize>,
        wait: Option<Duration>,
    ) -> std::result::Result<PreparedText, RouterError> {
        let queued = Instant::now();
        let slot = self
            .slot(model)
            .ok_or_else(|| RouterError::ModelNotFound(model.to_owned()))?;
        self.touch(model);
        let prompt_tokens = slot.count_tokens(&prompt)?;
        let context_len = slot.manifest().context_len;
        if prompt_tokens >= context_len {
            return Err(RouterError::ContextOverflow {
                tokens: prompt_tokens,
                limit: context_len,
            });
        }
        let max_tokens = requested.unwrap_or(16).min(context_len - prompt_tokens);
        if max_tokens == 0 {
            return Err(RouterError::InvalidMaxTokens);
        }
        let permit = match wait {
            Some(wait) => slot.acquire(wait).await.map_err(RouterError::from)?,
            None => slot.try_acquire().ok_or(RouterError::Busy)?,
        };
        self.metrics.queue(&slot.manifest().id, queued.elapsed());
        let session_id = next_session_id();
        let cancel = CancelToken::new();
        let model = slot.manifest().id.clone();
        let guard = self.register_for(&session_id, cancel.clone(), Some(model.clone()))?;
        self.metrics.begin(&model, prompt_tokens);
        Ok(PreparedText {
            slot,
            permit,
            session_id,
            prompt,
            images: Vec::new(),
            prompt_tokens,
            max_tokens,
            cancel,
            guard,
            started: Instant::now(),
        })
    }

    #[must_use]
    pub fn queue_timeout(&self) -> Duration {
        Duration::from_millis(self.queue_timeout_ms.load(Ordering::SeqCst))
    }

    pub fn set_queue_timeout(&self, timeout: Duration) {
        self.queue_timeout_ms.store(
            u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            Ordering::SeqCst,
        );
    }

    pub fn set_max_loaded(&self, max_loaded: usize) {
        self.max_loaded.store(max_loaded, Ordering::SeqCst);
        self.evict();
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    /// Register a request-scoped cancellation token.
    pub fn register(&self, id: &str, token: CancelToken) -> Result<CancelGuard> {
        self.register_for(id, token, None)
    }

    fn register_for(
        &self,
        id: &str,
        token: CancelToken,
        model: Option<String>,
    ) -> Result<CancelGuard> {
        self.cancels
            .lock()
            .map_err(|_| EngineError::Other("cancel table poisoned".into()))?
            .insert(
                id.to_owned(),
                CancelEntry {
                    token: token.clone(),
                    model,
                },
            );
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
        let entry = self
            .cancels
            .lock()
            .ok()
            .and_then(|guard| guard.get(id).cloned());
        if let Some(entry) = entry {
            entry.token.cancel();
            if let Some(model) = entry.model {
                self.metrics.cancelled(&model);
            }
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn record_completion(&self, model: &str, completion_tokens: usize) {
        self.metrics.completion(model, completion_tokens);
    }

    pub fn record_ttft(&self, model: &str, elapsed: Duration) {
        self.metrics.ttft(model, elapsed);
    }

    pub fn record_cancellation(&self, model: &str) {
        self.metrics.cancelled(model);
    }

    pub fn record_asr(&self, model: &str, audio_seconds: f64, elapsed: Duration) {
        self.metrics.asr(model, audio_seconds, elapsed);
    }

    pub fn record_generation(&self, model: &str, elapsed: Duration) {
        self.metrics.generation(model, elapsed);
    }

    pub fn record_timeout(&self, model: &str) {
        self.metrics.timeout(model);
    }

    #[must_use]
    pub fn active_guard(&self) -> ActiveGuard {
        ActiveGuard::new(Arc::clone(&self.active))
    }
}

/// Cancels and unregisters a request when the handler is dropped.
pub struct CancelGuard {
    cancels: CancelTable,
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

fn next_session_id() -> String {
    let counter = NEXT_SESSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    format!("sess-{seconds:x}-{counter:x}")
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
