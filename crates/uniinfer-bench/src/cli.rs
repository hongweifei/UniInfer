use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "uniinfer-bench", about = "Text scenario benchmarks")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
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
