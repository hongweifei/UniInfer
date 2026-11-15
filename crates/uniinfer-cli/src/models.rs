//! Model inspection/import commands.

use std::path::PathBuf;

use anyhow::{Context, Result};
use uniinfer_config::{ConfigHandle, ModelEntry};
use uniinfer_model_inspect::{InspectOptions, inspect};

pub(crate) fn inspect_model(path: PathBuf, id: Option<String>) -> Result<()> {
    let inspection = inspect(
        &path,
        &InspectOptions {
            id,
            ..InspectOptions::default()
        },
    )
    .with_context(|| format!("inspect {}", path.display()))?;
    for warning in &inspection.warnings {
        eprintln!("warning: {warning}");
    }
    println!("{}", inspection.to_toml()?);
    Ok(())
}

pub(crate) fn import_model(
    path: PathBuf,
    config_path: Option<PathBuf>,
    id: Option<String>,
) -> Result<()> {
    let inspection = inspect(
        &path,
        &InspectOptions {
            id,
            ..InspectOptions::default()
        },
    )
    .with_context(|| format!("inspect {}", path.display()))?;
    for warning in &inspection.warnings {
        eprintln!("warning: {warning}");
    }
    let manifest = inspection.write_manifest()?;
    let Some(config_path) = config_path else {
        println!("{}", manifest.display());
        println!("no --config/UNIINFER_CONFIG; add this manifest to a config manually");
        return Ok(());
    };
    let handle = ConfigHandle::load(config_path.clone())?;
    let mut config = handle.snapshot()?;
    let base = handle.base_dir();
    let manifest = manifest
        .strip_prefix(&base)
        .map_or(manifest.clone(), std::path::Path::to_path_buf);
    if !config.models.iter().any(|model| model.manifest == manifest) {
        config.models.push(ModelEntry::new(manifest.clone()));
    }
    config.validate()?;
    handle.replace(config)?;
    println!("{}", manifest.display());
    Ok(())
}
