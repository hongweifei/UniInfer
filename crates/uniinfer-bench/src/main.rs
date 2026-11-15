mod cli;
mod compare;
mod measure;
mod sys;

use clap::Parser;

use cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Run {
            manifest,
            prompt,
            max_tokens,
            iterations,
            save,
        } => measure::run(&manifest, &prompt, max_tokens, iterations, save.as_deref()),
        Command::Compare {
            baseline,
            current,
            warn,
            fail,
        } => compare::compare(&baseline, &current, warn, fail),
    }
}
