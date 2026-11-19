//! ONNX implementation adapters.
//!
//! The engine validates format/device/quantization and delegates the
//! model-family-specific loading to one adapter. New ONNX model families
//! should be added here, never as branches in the request path.

use ort::environment::Environment;
use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result};

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    match spec.manifest.implementation.as_str() {
        "onnx.graph" | "" => crate::generic::load(environment, spec, class),
        "onnx.pipeline" => crate::pipeline::load(environment, spec, class),
        "onnx.layered" => crate::layered::load(environment, spec, class),
        "onnx.diffusion" => crate::image::load(environment, spec, class),
        other => Err(EngineError::UnsupportedEngine(format!(
            "ONNX implementation {other:?} is not registered"
        ))),
    }
}
