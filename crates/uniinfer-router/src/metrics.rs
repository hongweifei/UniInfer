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
    pub asr_requests: u64,
    pub asr_audio_seconds: f64,
    pub asr_compute_micros: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MetricsSnapshot {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_micros: u64,
    pub ttft_samples: u64,
    pub cancellations: u64,
    pub asr_requests: u64,
    pub asr_audio_seconds: f64,
    pub asr_compute_micros: u64,
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
    asr_requests: AtomicU64,
    asr_audio_micros: AtomicU64,
    asr_compute_micros: AtomicU64,
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

    pub fn asr(&self, model: &str, audio_seconds: f64, elapsed: Duration) {
        let audio_micros = (audio_seconds.max(0.0) * 1_000_000.0) as u64;
        let compute_micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
        self.asr_requests.fetch_add(1, Ordering::Relaxed);
        self.asr_audio_micros
            .fetch_add(audio_micros, Ordering::Relaxed);
        self.asr_compute_micros
            .fetch_add(compute_micros, Ordering::Relaxed);
        if let Ok(mut models) = self.models.lock() {
            let entry = models.entry(model.to_owned()).or_default();
            entry.asr_requests += 1;
            entry.asr_audio_seconds += audio_seconds.max(0.0);
            entry.asr_compute_micros += compute_micros;
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
            asr_requests: self.asr_requests.load(Ordering::Relaxed),
            asr_audio_seconds: self.asr_audio_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            asr_compute_micros: self.asr_compute_micros.load(Ordering::Relaxed),
            models,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_asr_audio_and_compute_time() {
        let metrics = Metrics::default();
        metrics.asr("whisper", 2.0, Duration::from_millis(500));
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.asr_requests, 1);
        assert!((snapshot.asr_audio_seconds - 2.0).abs() < 1e-9);
        assert_eq!(snapshot.asr_compute_micros, 500_000);
        assert_eq!(snapshot.models["whisper"].asr_requests, 1);
    }
}
