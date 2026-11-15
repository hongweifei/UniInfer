//! Qwen2 family adapter.

use candle_core::Tensor;
use uniinfer_core::{ModelFormat, Result};

use super::{LoadContext, TextFamily, TextModel};

mod model;
use model::Model;

pub(crate) struct Qwen2Family;

impl TextFamily for Qwen2Family {
    fn name(&self) -> &'static str {
        "qwen2"
    }

    fn load(&self, ctx: &LoadContext<'_>) -> Result<Box<dyn TextModel>> {
        let model = match ctx.format {
            ModelFormat::Gguf => model::load_gguf(
                ctx.model_paths.first().ok_or_else(|| {
                    uniinfer_core::EngineError::ManifestInvalid("files is empty".into())
                })?,
                ctx.device,
                ctx.cancel,
            )?,
            ModelFormat::Safetensors => model::load_safetensors(
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

struct Qwen2Model(Model);

impl TextModel for Qwen2Model {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        self.0.forward(input, offset)
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}
