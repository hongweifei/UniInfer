use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use uniinfer_service::{CancelToken, Runtime};

use crate::sys::{current_process_memory, git_commit, total_memory, unix_seconds};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Measurement {
    pub(crate) schema: u32,
    pub(crate) created_unix: u64,
    pub(crate) model: String,
    pub(crate) engine: String,
    pub(crate) device: String,
    pub(crate) ep: String,
    pub(crate) commit: String,
    pub(crate) ram_bytes: u64,
    pub(crate) prompt_bytes: usize,
    max_tokens: usize,
    iterations: usize,
    pub(crate) generated_bytes: usize,
    pub(crate) ttft_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) tokens_per_second: f64,
    pub(crate) peak_rss_bytes: u64,
}

pub(crate) fn run(
    manifest_path: &Path,
    prompt: &str,
    max_tokens: usize,
    iterations: usize,
    save: Option<&Path>,
) -> Result<()> {
    if iterations == 0 {
        bail!("iterations must be at least 1");
    }

    let runtime = Runtime::new()?;
    let slot = runtime.load(manifest_path)?;
    let manifest = slot.manifest();

    // Warm up caches and the ONNX graph.
    slot.generate(prompt, max_tokens, &CancelToken::new(), &mut |_| true)?;

    let mut ttft_total = 0.0;
    let mut total_total = 0.0;
    let mut generated = 0usize;
    let mut generated_tokens = 0usize;

    for _ in 0..iterations {
        let token = CancelToken::new();
        let start = Instant::now();
        let mut first_byte = None;
        let mut bytes = 0usize;
        let output = slot.generate(prompt, max_tokens, &token, &mut |chunk| {
            if first_byte.is_none() {
                first_byte = Some(start.elapsed());
            }
            bytes += chunk.len();
            true
        })?;
        generated_tokens += output.tokens;
        let elapsed = start.elapsed();
        ttft_total += first_byte.unwrap_or(elapsed).as_secs_f64() * 1000.0;
        total_total += elapsed.as_secs_f64() * 1000.0;
        generated += bytes;
    }

    let ttft_ms = ttft_total / iterations as f64;
    let total_ms = total_total / iterations as f64;
    let decode_seconds = (total_ms - ttft_ms).max(0.001) / 1000.0;
    let measurement = Measurement {
        schema: 1,
        created_unix: unix_seconds(),
        model: manifest.id.clone(),
        engine: slot.engine_name().to_owned(),
        device: slot
            .placement()
            .devices
            .first()
            .map_or_else(|| "unknown".into(), |device| device.id.clone()),
        ep: slot
            .placement()
            .devices
            .first()
            .map_or_else(|| "unknown".into(), |device| device.backend.clone()),
        commit: git_commit(),
        ram_bytes: total_memory(),
        prompt_bytes: prompt.len(),
        max_tokens,
        iterations,
        generated_bytes: generated,
        ttft_ms,
        total_ms,
        tokens_per_second: (generated_tokens as f64 / iterations as f64) / decode_seconds,
        peak_rss_bytes: current_process_memory(),
    };

    let json = serde_json::to_string_pretty(&measurement)?;
    println!("{json}");
    if let Some(path) = save {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write baseline {}", path.display()))?;
    }
    Ok(())
}
