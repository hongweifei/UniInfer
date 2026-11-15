//! Model adapters for ONNX speech models.
//!
//! The ONNX engine only owns graph execution. An ASR adapter is selected from
//! the manifest `[audio].architecture` field, so adding another model family
//! does not change `OnnxEngine` or the API layer.

use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result};

mod ctc;
mod mel;
mod whisper;

pub(crate) fn load(
    environment: &ort::environment::Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let audio = spec
        .manifest
        .audio
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("missing [audio] config".into()))?;
    match audio.architecture.as_deref() {
        Some("whisper") => Ok(Box::new(whisper::load(environment, spec, class)?)),
        Some("ctc") => Ok(Box::new(ctc::load(environment, spec, class)?)),
        Some(other) => Err(EngineError::UnsupportedEngine(format!(
            "ONNX speech architecture {other:?} is not registered"
        ))),
        None => Err(EngineError::ManifestInvalid(
            "audio.architecture is required for a multi-graph speech model".into(),
        )),
    }
}
