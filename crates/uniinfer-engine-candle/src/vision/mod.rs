//! Vision-language family registry.

mod preprocess;
mod qwen3_vl;

use std::path::Path;

use candle_core::{DType, Device};
use uniinfer_core::{EngineError, EngineLoadSpec, EngineSession, ModelFormat, Result};

pub(crate) use preprocess::ImageProcessor;

pub(crate) fn load(
    spec: &EngineLoadSpec,
    family: &str,
    dtype: DType,
    device: &Device,
    config_path: &Path,
    model_paths: &[std::path::PathBuf],
) -> Result<Box<dyn EngineSession>> {
    match family {
        "qwen3_vl" => qwen3_vl::load(spec, dtype, device, config_path, model_paths),
        other => Err(EngineError::UnsupportedEngine(format!(
            "candle vision family {other:?}"
        ))),
    }
}

pub(crate) fn verify_safetensors(format: ModelFormat) -> Result<()> {
    if format != ModelFormat::Safetensors {
        return Err(EngineError::UnsupportedFormat(format));
    }
    Ok(())
}
