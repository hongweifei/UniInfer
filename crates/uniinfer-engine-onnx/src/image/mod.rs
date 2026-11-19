//! Image-generation pipelines.
//!
//! The engine owns graph loading and cancellation; each architecture adapter
//! owns prompt encoding, latent layout, scheduler semantics and graph bindings.

mod scheduler;
mod sd;

use ort::environment::Environment;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, ImageArchitecture, Result,
};

/// Load the adapter selected by the manifest's image architecture.
pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let architecture = spec
        .manifest
        .image
        .as_ref()
        .and_then(|image| image.diffusion.as_ref())
        .map(|diffusion| diffusion.architecture)
        .ok_or_else(|| EngineError::ManifestInvalid("image.diffusion config required".into()))?;
    match architecture {
        ImageArchitecture::Sd => sd::load(environment, spec, class),
        ImageArchitecture::Flux => Err(EngineError::UnsupportedEngine(
            "Flux image adapter is not available in this build".into(),
        )),
        ImageArchitecture::QwenImage => Err(EngineError::UnsupportedEngine(
            "Qwen-Image adapter is not available in this build".into(),
        )),
        ImageArchitecture::ZImage => Err(EngineError::UnsupportedEngine(
            "Z-Image adapter is not available in this build".into(),
        )),
        ImageArchitecture::Generic => Err(EngineError::UnsupportedEngine(
            "generic image adapter is not available in this build".into(),
        )),
    }
}
