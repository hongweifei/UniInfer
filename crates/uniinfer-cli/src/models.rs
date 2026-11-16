//! Model inspection/import commands.

use std::path::PathBuf;

use anyhow::{Context, Result};
use uniinfer_model_inspect::{InspectOptions, inspect};
use uniinfer_service::Runtime;

pub(crate) fn inspect_model(
    path: PathBuf,
    id: Option<String>,
    check: bool,
    load: bool,
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
    let runtime = (check || load)
        .then(Runtime::new)
        .transpose()
        .context("initialize runtime")?;
    if check {
        let runtime = runtime.as_ref().expect("runtime initialized");
        match runtime.probe_manifest(&inspection.manifest) {
            Ok(route) => eprintln!(
                "probe: supported engine={} devices={:?}",
                route.engine_name,
                route
                    .placement
                    .devices
                    .iter()
                    .map(|device| device.class)
                    .collect::<Vec<_>>()
            ),
            Err(error) => {
                eprintln!("probe: unsupported {}: {error}", error.code());
                return Err(error).context("probe generated manifest");
            }
        }
    }
    if load {
        let runtime = runtime.as_ref().expect("runtime initialized");
        let audio_only = inspection.manifest.audio.is_some() && inspection.manifest.text.is_none();
        if audio_only {
            runtime
                .load_audio_manifest(inspection.manifest.clone(), &inspection.root)
                .context("load generated audio manifest")?;
        } else {
            runtime
                .load_manifest(inspection.manifest.clone(), &inspection.root)
                .context("load generated text manifest")?;
        }
        eprintln!("load: ok");
    }
    println!("{}", inspection.to_toml()?);
    Ok(())
}

pub(crate) fn import_model(path: PathBuf, id: Option<String>) -> Result<()> {
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
    println!("{}", manifest.display());
    Ok(())
}
