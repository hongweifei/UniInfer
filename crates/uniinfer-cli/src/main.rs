use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use uniinfer_service::{CancelToken, Runtime};

mod models;
mod serve;

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
enum ModelsCommand {
    /// Print a generated manifest without writing it.
    Inspect {
        path: PathBuf,
        #[arg(long)]
        id: Option<String>,
    },
    /// Generate manifest.toml and add it to the application config.
    Import {
        path: PathBuf,
        #[arg(long)]
        id: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List discovered devices and execution backends.
    Devices,
    /// Inspect or import a local model directory.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
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
        Command::Models { command } => match command {
            ModelsCommand::Inspect { path, id } => models::inspect_model(path, id),
            ModelsCommand::Import { path, id } => {
                models::import_model(path, config_path.clone(), id)
            }
        },
        Command::Serve {
            manifests,
            addr,
            queue_timeout_ms,
            token,
        } => serve::serve(config_path, manifests, addr, queue_timeout_ms, token).await,
        Command::Run {
            manifest,
            prompt,
            max_tokens,
        } => run(&manifest, &prompt, max_tokens),
    }
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
