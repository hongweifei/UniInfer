//! Text model family registry.
//!
//! The engine never checks a concrete family; it resolves the manifest's
//! `text.family` through this registry and lets the family own loading and
//! forward semantics.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use uniinfer_core::{CancelToken, ModelFormat, Quantization, Result};

pub(crate) mod qwen2;
pub(crate) mod qwen3;

pub(crate) const GGUF_QUANTIZATIONS: &[Quantization] = &[
    Quantization::Q4_0,
    Quantization::Q4_1,
    Quantization::Q5_0,
    Quantization::Q5_1,
    Quantization::Q8_0,
    Quantization::Q2K,
    Quantization::Q3KS,
    Quantization::Q3KM,
    Quantization::Q3KL,
    Quantization::Q4KS,
    Quantization::Q4KM,
    Quantization::Q5KS,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8K,
    Quantization::Q2KS,
];
pub(crate) const SAFETENSORS_QUANTIZATIONS: &[Quantization] =
    &[Quantization::Bf16, Quantization::Fp16, Quantization::Fp32];

pub(crate) trait TextModel: Send {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor>;
    fn clear(&mut self);
}

pub(crate) struct LoadContext<'a> {
    pub(crate) format: ModelFormat,
    pub(crate) model_paths: &'a [std::path::PathBuf],
    pub(crate) config_path: &'a Path,
    pub(crate) dtype: DType,
    pub(crate) device: &'a Device,
    pub(crate) cancel: &'a CancelToken,
}

pub(crate) trait TextFamily: Send + Sync {
    fn name(&self) -> &'static str;
    fn load(&self, ctx: &LoadContext<'_>) -> Result<Box<dyn TextModel>>;
}

#[derive(Default)]
pub(crate) struct TextRegistry {
    families: HashMap<&'static str, Arc<dyn TextFamily>>,
}

impl TextRegistry {
    pub(crate) fn new() -> Self {
        let mut registry = Self::default();
        registry.register(Arc::new(qwen3::Qwen3Family));
        registry.register(Arc::new(qwen2::Qwen2Family));
        registry
    }

    fn register(&mut self, family: Arc<dyn TextFamily>) {
        self.families.insert(family.name(), family);
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<dyn TextFamily>> {
        self.families.get(name).cloned()
    }

    #[cfg(test)]
    pub(crate) fn supported(&self, name: &str) -> bool {
        self.families.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_known_families() {
        let registry = TextRegistry::new();
        assert!(registry.supported("qwen3"));
        assert!(registry.supported("qwen2"));
        assert!(!registry.supported("unknown-family"));
    }
}
