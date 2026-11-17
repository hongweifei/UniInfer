//! Model-load timeout and memory-pressure retry policy.

use std::future::Future;
use std::time::{Duration, Instant};

use tokio::task::JoinHandle;
use uniinfer_core::{CancelToken, EngineError, Result};

const MAX_MEMORY_LOAD_RETRIES: usize = 8;

pub(super) async fn await_load<T>(
    timeout: Option<Duration>,
    task: JoinHandle<Result<T>>,
    cancel: CancelToken,
) -> Result<T> {
    let Some(timeout) = timeout else {
        return task
            .await
            .map_err(|error| EngineError::Other(format!("model load task: {error}")))?;
    };
    let started = Instant::now();
    let joined = match tokio::time::timeout(timeout, task).await {
        Ok(joined) => joined,
        Err(_) => {
            cancel.cancel();
            return Err(EngineError::Timeout);
        }
    };
    let loaded =
        joined.map_err(|error| EngineError::Other(format!("model load task: {error}")))??;
    // The timer has ~1ms granularity, so a load can finish after the budget
    // but before the timer fires. Discard it as timed out too.
    if started.elapsed() > timeout {
        cancel.cancel();
        return Err(EngineError::Timeout);
    }
    Ok(loaded)
}

pub(super) async fn retry_memory<T, F, Fut, E>(id: &str, mut load: F, mut evict: E) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
    E: FnMut(&str) -> bool,
{
    let mut attempts = 0;
    loop {
        match load().await {
            Ok(value) => return Ok(value),
            Err(error)
                if is_memory_error(&error) && attempts < MAX_MEMORY_LOAD_RETRIES && evict(id) =>
            {
                attempts += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

fn is_memory_error(error: &EngineError) -> bool {
    matches!(
        error,
        EngineError::InsufficientVram { .. } | EngineError::InsufficientMemory { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn classification_only_targets_memory_errors() {
        assert!(is_memory_error(&EngineError::InsufficientVram {
            required: "1".into(),
            available: "0".into(),
        }));
        assert!(is_memory_error(&EngineError::InsufficientMemory {
            required: "1".into(),
            available: "0".into(),
        }));
        assert!(!is_memory_error(&EngineError::Timeout));
    }

    #[tokio::test]
    async fn retries_until_eviction_stops_making_progress() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let outcome = retry_memory(
            "model",
            move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>(EngineError::InsufficientVram {
                        required: "1".into(),
                        available: "0".into(),
                    })
                }
            },
            |_| false,
        )
        .await;
        assert!(outcome.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
