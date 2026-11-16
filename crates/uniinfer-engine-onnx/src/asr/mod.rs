//! Model adapters for ONNX speech models.
//!
//! The ONNX engine only owns graph execution. An ASR adapter is selected from
//! the manifest `[audio].architecture` field, so adding another model family
//! does not change `OnnxEngine` or the API layer.

use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result};

mod ctc;
mod decode;
mod frontend;
mod options;
mod whisper;

pub(crate) fn load(
    environment: &ort::environment::Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    if spec.cancel.is_cancelled() {
        return Err(EngineError::Cancelled);
    }
    let audio = spec
        .manifest
        .audio
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("missing [audio] config".into()))?;
    match audio.family.as_deref() {
        Some("whisper") => Ok(Box::new(whisper::load(environment, spec, class)?)),
        Some("ctc") => Ok(Box::new(ctc::load(environment, spec, class)?)),
        Some(other) => Err(EngineError::UnsupportedEngine(format!(
            "ONNX speech architecture {other:?} is not registered"
        ))),
        None => Err(EngineError::ManifestInvalid(
            "audio.family is required for a multi-graph speech model".into(),
        )),
    }
}
