//! Qwen3 family adapter.

use candle_core::Tensor;
use uniinfer_core::{EngineError, ModelFormat, Result};

mod model;

use super::{LoadContext, TextFamily, TextModel};

pub(crate) struct Qwen3Family;

impl TextFamily for Qwen3Family {
    fn name(&self) -> &'static str {
        "qwen3"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Box<dyn TextModel>> {
        let model = match ctx.format {
            ModelFormat::Gguf => {
                let path = ctx
                    .model_paths
                    .first()
                    .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
                model::load_gguf(path, ctx.device, ctx.cancel)?
            }
            ModelFormat::Safetensors => model::load_safetensors(
                ctx.model_paths,
                ctx.config_path,
                ctx.dtype,
                ctx.device,
                ctx.cancel,
            )?,
            _ => unreachable!(),
        };
        Ok(Box::new(Qwen3Model(model)))
    }
}

struct Qwen3Model(model::Model);

impl TextModel for Qwen3Model {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        self.0.forward(input, offset)
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}
