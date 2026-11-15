use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use uniinfer_api::ApiConfig;
use uniinfer_config::{ConfigHandle, ModelEntry, UniInferConfig};
use uniinfer_service::{CancelToken, Runtime};

#[derive(Debug, Parser)]
#[command(name = "uniinfer", version, about = "UniInfer local inference runtime")]
struct Cli {
    /// Durable TOML/JSON configuration used by serve; GUI edits the same file.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List discovered devices and execution backends.
    Devices,
    /// Serve the OpenAI-compatible HTTP API.
    Serve {
        /// Model manifest paths; when set, they replace config.models.
        #[arg(long = "manifest")]
        manifests: Vec<PathBuf>,
        /// Override server.addr from the config file.
        #[arg(long)]
        addr: Option<std::net::SocketAddr>,
        /// Override server.queue_timeout_ms from the config file.
        #[arg(long)]
        queue_timeout_ms: Option<u64>,
        /// Override server.token from the config file.
        #[arg(long)]
        token: Option<String>,
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

    let cli = Cli::parse();
    let config_path = cli
        .config
        .clone()
        .or_else(|| std::env::var_os("UNIINFER_CONFIG").map(std::path::PathBuf::from));
    match cli.command {
        Command::Devices => devices(),
        Command::Serve {
            manifests,
            addr,
            queue_timeout_ms,
            token,
        } => serve(config_path, manifests, addr, queue_timeout_ms, token).await,
        Command::Run {
            manifest,
            prompt,
            max_tokens,
        } => run(&manifest, &prompt, max_tokens),
    }
}

async fn serve(
    config_path: Option<PathBuf>,
    manifests: Vec<PathBuf>,
    addr: Option<std::net::SocketAddr>,
    queue_timeout_ms: Option<u64>,
    token: Option<String>,
) -> Result<()> {
    let source = match &config_path {
        Some(path) => ConfigHandle::load(path.clone())?,
        None => ConfigHandle::new(UniInferConfig::default(), None),
    };
    let mut config = source.snapshot()?;
    config.apply_env()?;
    if !manifests.is_empty() {
        config.models = manifests.into_iter().map(ModelEntry::new).collect();
    }
    if let Some(addr) = addr {
        config.server.addr = addr;
    }
    if let Some(queue_timeout_ms) = queue_timeout_ms {
        config.server.queue_timeout_ms = queue_timeout_ms;
    }
    if let Some(token) = token {
        config.server.token = Some(token);
    }
    config.validate_for_serve()?;
    let addr = config.server.addr;
    let handle = Arc::new(ConfigHandle::new(config, config_path));
    let api = ApiConfig::from_config(handle)?;
    uniinfer_api::serve(api, addr)
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

fn run(manifest_path: &Path, prompt: &str, max_tokens: usize) -> Result<()> {
    let runtime = Runtime::new().context("initialize runtime")?;
    let slot = runtime
        .load(manifest_path)
        .with_context(|| format!("load {}", manifest_path.display()))?;

    let mut stdout = std::io::stdout().lock();
    slot.generate(prompt, max_tokens, &CancelToken::new(), &mut |chunk| {
        stdout.write_all(chunk).is_ok() && stdout.flush().is_ok()
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
