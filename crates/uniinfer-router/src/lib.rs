//! Model catalog and session lifecycle shared by HTTP, WebSocket and CLI.
//!
//! The router owns the loaded model slots, the cancellation table and the
//! active-session gauge. Transport crates keep DTO parsing and wire formats.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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
    slots: Arc<HashMap<String, Arc<ModelSlot>>>,
    cancels: CancelTable,
    active: Arc<AtomicUsize>,
    metrics: metrics::Metrics,
    queue_timeout: Duration,
}

impl ModelRouter {
    #[must_use]
    pub fn new(slots: HashMap<String, Arc<ModelSlot>>, queue_timeout: Duration) -> Self {
        Self {
            slots: Arc::new(slots),
            cancels: Arc::new(Mutex::new(HashMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
            metrics: metrics::Metrics::default(),
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

    /// Resolve a model, validate context, acquire its permit and register
    /// cancellation. Waits up to the router's queue timeout.
    pub async fn prepare_text(
        &self,
        model: &str,
        prompt: String,
        requested: Option<usize>,
    ) -> std::result::Result<PreparedText, RouterError> {
        self.prepare_text_with(model, prompt, requested, Some(self.queue_timeout))
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
        let slot = self
            .slot(model)
            .ok_or_else(|| RouterError::ModelNotFound(model.to_owned()))?;
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
            prompt_tokens,
            max_tokens,
            cancel,
            guard,
            started: Instant::now(),
        })
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
