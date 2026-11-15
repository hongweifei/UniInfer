//! Qwen2 family adapter.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use uniinfer_core::{CancelToken, EngineError, ModelFormat, Result};

use super::{LoadContext, TextFamily, TextModel};

pub(crate) struct Qwen2Family;

impl TextFamily for Qwen2Family {
    fn name(&self) -> &'static str {
        "qwen2"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Box<dyn TextModel>> {
        let model = match ctx.format {
            ModelFormat::Gguf => {
                let path = ctx
                    .model_paths
                    .first()
                    .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
                load_gguf(path, ctx.device, ctx.cancel)?
            }
            ModelFormat::Safetensors => load_safetensors(
                ctx.model_paths,
                ctx.config_path,
                ctx.dtype,
                ctx.device,
                ctx.cancel,
            )?,
            _ => unreachable!(),
        };
        Ok(Box::new(Qwen2Model(model)))
    }
}

enum Model {
    Quantized(ModelWeights),
    Dense(ModelForCausalLM),
}

impl TextModel for Qwen2Model {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match &mut self.0 {
            Model::Quantized(model) => model.forward(input, offset),
            Model::Dense(model) => model.forward(input, offset),
        }
    }

    fn clear(&mut self) {
        match &mut self.0 {
            Model::Quantized(model) => model.clear_kv_cache(),
            Model::Dense(model) => model.clear_kv_cache(),
        }
    }
}

struct Qwen2Model(Model);

fn load_gguf(path: &Path, device: &candle_core::Device, cancel: &CancelToken) -> Result<Model> {
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
fn load_safetensors(
    paths: &[std::path::PathBuf],
    config_path: &Path,
    dtype: DType,
    device: &candle_core::Device,
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
