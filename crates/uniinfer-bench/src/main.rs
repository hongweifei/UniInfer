use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use uniinfer_backends::{GenerateOptions, generate_bytes_cancellable};
use uniinfer_core::CancelToken;
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;
use uniinfer_registry::EngineRegistry;

#[derive(Debug, Parser)]
#[command(name = "uniinfer-bench", about = "Text scenario benchmarks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Measure one model and optionally save the baseline.
    Run {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value = "hello")]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        max_tokens: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        save: Option<PathBuf>,
    },
    /// Compare a measurement against a baseline.
    Compare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        current: PathBuf,
        /// Degradation ratio that prints a warning.
        #[arg(long, default_value_t = 0.10)]
        warn: f64,
        /// Degradation ratio that fails the process.
        #[arg(long, default_value_t = 0.25)]
        fail: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Measurement {
    schema: u32,
    created_unix: u64,
    model: String,
    engine: String,
    device: String,
    prompt_bytes: usize,
    max_tokens: usize,
    iterations: usize,
    generated_bytes: usize,
    ttft_ms: f64,
    total_ms: f64,
    tokens_per_second: f64,
    peak_rss_bytes: u64,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            manifest,
            prompt,
            max_tokens,
            iterations,
            save,
        } => run(&manifest, &prompt, max_tokens, iterations, save.as_deref()),
        Command::Compare {
            baseline,
            current,
            warn,
            fail,
        } => compare(&baseline, &current, warn, fail),
    }
}

fn run(
    manifest_path: &Path,
    prompt: &str,
    max_tokens: usize,
    iterations: usize,
    save: Option<&Path>,
) -> Result<()> {
    if iterations == 0 {
        bail!("iterations must be at least 1");
    }

    let engine = OnnxEngine::new()?;
    let mut inventory = DeviceInventory::probe();
    for device in engine.probe_devices() {
        inventory.register(device);
    }
    let mut registry = EngineRegistry::new(inventory);
    registry.register(Arc::new(engine));
    let lease = registry.load(manifest_path)?;
    let manifest = lease.manifest();
    let tokenizer = manifest.tokenizer.as_deref().unwrap_or_default();
    let text = manifest
        .text
        .as_ref()
        .context("manifest has no [text] section")?;
    if tokenizer != "bytes" {
        bail!("bench supports byte tokenizers, found {tokenizer:?}");
    }

    let options = GenerateOptions {
        max_tokens,
        eos_token: text.eos_token,
    };

    // Warm up caches and the ONNX graph.
    generate_bytes_cancellable(
        lease.session(),
        prompt,
        text.vocab_size,
        manifest.context_len,
        &options,
        &CancelToken::new(),
        &mut |_| true,
    )?;

    let mut ttft_total = 0.0;
    let mut total_total = 0.0;
    let mut generated = 0usize;

    for _ in 0..iterations {
        let token = CancelToken::new();
        let start = Instant::now();
        let mut first_byte = None;
        let mut bytes = 0usize;
        generate_bytes_cancellable(
            lease.session(),
            prompt,
            text.vocab_size,
            manifest.context_len,
            &options,
            &token,
            &mut |_| {
                if first_byte.is_none() {
                    first_byte = Some(start.elapsed());
                }
                bytes += 1;
                true
            },
        )?;
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
        engine: lease.engine_name().to_owned(),
        device: lease
            .placement()
            .devices
            .first()
            .map_or_else(|| "unknown".into(), |device| device.id.clone()),
        prompt_bytes: prompt.len(),
        max_tokens,
        iterations,
        generated_bytes: generated,
        ttft_ms,
        total_ms,
        tokens_per_second: (generated as f64 / iterations as f64) / decode_seconds,
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

fn compare(baseline_path: &Path, current_path: &Path, warn: f64, fail: f64) -> Result<()> {
    let baseline: Measurement = serde_json::from_str(
        &std::fs::read_to_string(baseline_path)
            .with_context(|| format!("read {}", baseline_path.display()))?,
    )?;
    let current: Measurement = serde_json::from_str(
        &std::fs::read_to_string(current_path)
            .with_context(|| format!("read {}", current_path.display()))?,
    )?;

    let checks = [
        (
            "ttft_ms",
            ratio(current.ttft_ms, baseline.ttft_ms),
            true,
            "ms",
        ),
        (
            "total_ms",
            ratio(current.total_ms, baseline.total_ms),
            true,
            "ms",
        ),
        (
            "tokens_per_second",
            ratio(baseline.tokens_per_second, current.tokens_per_second),
            false,
            "tok/s",
        ),
        (
            "peak_rss_bytes",
            ratio(
                current.peak_rss_bytes as f64,
                baseline.peak_rss_bytes as f64,
            ),
            true,
            "bytes",
        ),
    ];

    let mut failed = false;
    for (name, degradation, lower_is_better, unit) in checks {
        if !degradation.is_finite() {
            continue;
        }
        let change = degradation * 100.0;
        let direction = if lower_is_better { "higher" } else { "lower" };
        if degradation > fail {
            println!("FAIL {name}: {change:.1}% {direction} is worse ({unit})");
            failed = true;
        } else if degradation > warn {
            println!("WARN {name}: {change:.1}% {direction} is worse ({unit})");
        } else {
            println!("ok   {name}: {change:+.1}%");
        }
    }

    if failed {
        bail!("benchmark regression exceeds {:.0}%", fail * 100.0);
    }
    Ok(())
}

fn ratio(current: f64, baseline: f64) -> f64 {
    if baseline <= 0.0 {
        f64::NAN
    } else {
        current / baseline - 1.0
    }
}

fn current_process_memory() -> u64 {
    let Ok(pid) = sysinfo::get_current_pid() else {
        return 0;
    };
    let Ok(mut system) = sysinfo::System::new() else {
        return 0;
    };
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map_or(0, sysinfo::Process::memory)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
