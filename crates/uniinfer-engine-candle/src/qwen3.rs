//! Qwen3 model loading and forward pass for Candle.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen3::ModelWeights;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use uniinfer_core::{EngineError, Quantization, Result};

pub(crate) const GGUF_QUANTIZATIONS: &[Quantization] = &[
    Quantization::Q4KM,
    Quantization::Q4_0,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8_0,
];
pub(crate) const SAFETENSORS_QUANTIZATIONS: &[Quantization] =
    &[Quantization::Bf16, Quantization::Fp16, Quantization::Fp32];

pub(crate) enum Model {
    Quantized(ModelWeights),
    Dense(ModelForCausalLM),
}

impl Model {
    pub(crate) fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Quantized(model) => model.forward(input, offset),
            Self::Dense(model) => model.forward(input, offset),
        }
    }

    pub(crate) fn clear(&mut self) {
        match self {
            Self::Quantized(model) => model.clear_kv_cache(),
            Self::Dense(model) => model.clear_kv_cache(),
        }
    }
}

pub(crate) fn load_gguf(path: &Path, device: &Device) -> Result<Model> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| EngineError::LoadFailed(format!("open {}: {error}", path.display())))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|error| EngineError::LoadFailed(format!("gguf metadata: {error}")))?;
    let model = ModelWeights::from_gguf(content, &mut file, device)
        .map_err(|error| EngineError::LoadFailed(format!("gguf weights: {error}")))?;
    Ok(Model::Quantized(model))
}

#[allow(unsafe_code)] // mmap of a user-supplied, checksum-verified weight file.
pub(crate) fn load_safetensors(
    path: &Path,
    config_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<Model> {
    let raw = std::fs::read_to_string(config_path)
        .map_err(|error| EngineError::LoadFailed(format!("config: {error}")))?;
    let config: Config = serde_json::from_str(&raw)
        .map_err(|error| EngineError::LoadFailed(format!("config json: {error}")))?;
    let builder = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
        .map_err(|error| EngineError::LoadFailed(format!("safetensors: {error}")))?;
    let model = ModelForCausalLM::new(&config, builder)
        .map_err(|error| EngineError::LoadFailed(format!("qwen3: {error}")))?;
    Ok(Model::Dense(model))
}
