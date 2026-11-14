use std::io::Write;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use uniinfer_api::ApiConfig;
use uniinfer_service::{CancelToken, Runtime};

#[derive(Debug, Parser)]
#[command(name = "uniinfer", version, about = "UniInfer local inference runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List discovered devices and execution backends.
    Devices,
    /// Serve the OpenAI-compatible HTTP API.
    Serve {
        /// Path to the model manifest.
        #[arg(long)]
        manifest: std::path::PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: std::net::SocketAddr,
        #[arg(long, default_value_t = 30_000)]
        queue_timeout_ms: u64,
    },
    /// Run one-shot greedy text generation.
    Run {
        /// Path to the model manifest.
        #[arg(long)]
        manifest: std::path::PathBuf,
        #[arg(long, default_value = "hello")]
        prompt: String,
        #[arg(long, default_value_t = 16)]
        max_tokens: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    match Cli::parse().command {
        Command::Devices => devices(),
        Command::Serve {
            manifest,
            addr,
            queue_timeout_ms,
        } => serve(manifest, addr, queue_timeout_ms).await,
        Command::Run {
            manifest,
            prompt,
            max_tokens,
        } => run(&manifest, &prompt, max_tokens),
    }
}

async fn serve(
    manifest: std::path::PathBuf,
    addr: std::net::SocketAddr,
    queue_timeout_ms: u64,
) -> Result<()> {
    let mut config = ApiConfig::new(manifest);
    config.queue_timeout = Duration::from_millis(queue_timeout_ms);
    uniinfer_api::serve(config, addr)
        .await
        .context("serve HTTP API")
}

fn devices() -> Result<()> {
    let runtime = Runtime::new().context("initialize runtime")?;
    println!(
        "{:<10} {:<16} {:<10} {:>12}",
        "ID", "CLASS", "BACKEND", "MEMORY"
    );
    for device in runtime.devices() {
        println!(
            "{:<10} {:<16} {:<10} {:>12}",
            device.id,
            device.class.to_string(),
            device.backend,
            format_bytes(device.memory_bytes)
        );
    }
    Ok(())
}

fn run(manifest_path: &std::path::Path, prompt: &str, max_tokens: usize) -> Result<()> {
    let runtime = Runtime::new().context("initialize runtime")?;
    let slot = runtime
        .load(manifest_path)
        .with_context(|| format!("load {}", manifest_path.display()))?;

    let mut stdout = std::io::stdout().lock();
    slot.generate(prompt, max_tokens, &CancelToken::new(), &mut |byte| {
        stdout.write_all(&[byte]).is_ok() && stdout.flush().is_ok()
    })?;
    writeln!(stdout)?;
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes == 0 {
        "unknown".into()
    } else {
        format!("{:.1} GiB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}
