//! Model adapters for ONNX speech models.
//!
//! The ONNX engine only owns graph execution. An ASR adapter is selected from
//! the pipeline adapter step, so adding another speech family stays in the
//! adapter layer.

use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result};

mod ctc;
mod decode;
mod frontend;
mod options;
mod whisper;

pub(crate) fn load_named(
    environment: &ort::environment::Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
    adapter: &str,
) -> Result<Box<dyn EngineSession>> {
    match adapter {
        "whisper" => Ok(Box::new(whisper::load(environment, spec, class)?)),
        "ctc" => Ok(Box::new(ctc::load(environment, spec, class)?)),
        other => Err(EngineError::UnsupportedEngine(format!(
            "ONNX speech adapter {other:?} is not registered"
        ))),
    }
}
