//! Qwen2 weight loading and forward pass.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use uniinfer_core::{CancelToken, EngineError, Result};

pub(super) enum Model {
    Quantized(ModelWeights),
    Dense(ModelForCausalLM),
}

impl Model {
    pub(super) fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Quantized(model) => model.forward(input, offset),
            Self::Dense(model) => model.forward(input, offset),
        }
    }

    pub(super) fn clear(&mut self) {
        match self {
            Self::Quantized(model) => model.clear_kv_cache(),
            Self::Dense(model) => model.clear_kv_cache(),
        }
    }
}

pub(super) fn load_gguf(path: &Path, device: &Device, cancel: &CancelToken) -> Result<Model> {
    let file = std::fs::File::open(path)
        .map_err(|error| EngineError::LoadFailed(format!("open {}: {error}", path.display())))?;
    let mut file = crate::cancel_io::CancellableReader::new(file, cancel.clone());
    let content = gguf_file::Content::read(&mut file).map_err(|error| {
        if cancel.is_cancelled() {
            EngineError::Cancelled
        } else {
            EngineError::LoadFailed(format!("gguf metadata: {error}"))
        }
    })?;
    let model = ModelWeights::from_gguf(content, &mut file, device).map_err(|error| {
        if cancel.is_cancelled() {
            EngineError::Cancelled
        } else {
            EngineError::LoadFailed(format!("qwen2 weights: {error}"))
        }
    })?;
    Ok(Model::Quantized(model))
}

#[allow(unsafe_code)] // mmap of a user-supplied, checksum-verified weight file.
pub(super) fn load_safetensors(
    paths: &[std::path::PathBuf],
    config_path: &Path,
    dtype: DType,
    device: &Device,
    cancel: &CancelToken,
) -> Result<Model> {
    if cancel.is_cancelled() {
        return Err(EngineError::Cancelled);
    }
    let raw = std::fs::read_to_string(config_path)
        .map_err(|error| EngineError::LoadFailed(format!("config: {error}")))?;
    let config: Config = serde_json::from_str(&raw)
        .map_err(|error| EngineError::LoadFailed(format!("qwen2 config: {error}")))?;
    let paths = paths
        .iter()
        .map(std::path::PathBuf::as_path)
        .collect::<Vec<_>>();
    let builder = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device) }
        .map_err(|error| EngineError::LoadFailed(format!("safetensors: {error}")))?;
    let model = ModelForCausalLM::new(&config, builder)
        .map_err(|error| EngineError::LoadFailed(format!("qwen2: {error}")))?;
    Ok(Model::Dense(model))
}
