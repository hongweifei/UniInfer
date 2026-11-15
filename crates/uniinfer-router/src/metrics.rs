//! Request, token and time-to-first-token counters shared by HTTP/WebSocket.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelMetrics {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_micros: u64,
    pub ttft_samples: u64,
    pub cancellations: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MetricsSnapshot {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_micros: u64,
    pub ttft_samples: u64,
    pub cancellations: u64,
    pub models: BTreeMap<String, ModelMetrics>,
}

#[derive(Default)]
pub struct Metrics {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    ttft_micros: AtomicU64,
    ttft_samples: AtomicU64,
    cancellations: AtomicU64,
    models: Mutex<BTreeMap<String, ModelMetrics>>,
}

impl Metrics {
    pub fn begin(&self, model: &str, prompt_tokens: usize) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        if let Ok(mut models) = self.models.lock() {
            let entry = models.entry(model.to_owned()).or_default();
            entry.requests += 1;
            entry.prompt_tokens += prompt_tokens as u64;
        }
    }

    pub fn completion(&self, model: &str, completion_tokens: usize) {
        self.completion_tokens
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        if let Ok(mut models) = self.models.lock() {
            models
                .entry(model.to_owned())
                .or_default()
                .completion_tokens += completion_tokens as u64;
        }
    }

    pub fn ttft(&self, model: &str, elapsed: Duration) {
        let micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        self.ttft_micros.fetch_add(micros, Ordering::Relaxed);
        self.ttft_samples.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut models) = self.models.lock() {
            let entry = models.entry(model.to_owned()).or_default();
            entry.ttft_micros += micros;
            entry.ttft_samples += 1;
        }
    }

    pub fn cancelled(&self, model: &str) {
        self.cancellations.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut models) = self.models.lock() {
            models.entry(model.to_owned()).or_default().cancellations += 1;
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let models = self
            .models
            .lock()
            .map(|models| models.clone())
            .unwrap_or_default();
        MetricsSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            ttft_micros: self.ttft_micros.load(Ordering::Relaxed),
            ttft_samples: self.ttft_samples.load(Ordering::Relaxed),
            cancellations: self.cancellations.load(Ordering::Relaxed),
            models,
        }
    }
}
