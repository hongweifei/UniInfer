use std::fs;
use std::path::Path;

use serde::Deserialize;
use uniinfer_core::{Modality, Quantization, TextConfig};

use crate::error::{InspectError, Result};

#[derive(Debug, Deserialize)]
pub(crate) struct ConfigJson {
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    #[serde(default)]
    pub(crate) model_type: Option<String>,
    #[serde(default)]
    pub(crate) vocab_size: Option<usize>,
    #[serde(default)]
    pub(crate) eos_token_id: Option<u32>,
    #[serde(default)]
    pub(crate) max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub(crate) max_source_positions: Option<usize>,
    #[serde(default)]
    pub(crate) max_length: Option<usize>,
    #[serde(default)]
    pub(crate) torch_dtype: Option<String>,
    #[serde(default)]
    pub(crate) quantization_config: Option<QuantizationConfig>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct QuantizationConfig {
    #[serde(default)]
    pub(crate) bits: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PreprocessorConfig {
    #[serde(default)]
    pub(crate) sampling_rate: Option<u32>,
}

pub(crate) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|error| InspectError::Config {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

pub(crate) fn text_config(
    root: &Path,
    config: Option<&ConfigJson>,
    tokenizer: Option<&Path>,
    config_path: Option<&Path>,
) -> Option<TextConfig> {
    let config = config?;
    let vocab_size = config.vocab_size?;
    Some(TextConfig {
        vocab_size,
        eos_token: config.eos_token_id,
        family: architecture_name(config),
        config_file: config_path.map(|path| crate::files::relative(root, path)),
        tokenizer_file: tokenizer.map(|path| crate::files::relative(root, path)),
    })
}

pub(crate) fn architecture_name(config: &ConfigJson) -> Option<String> {
    config
        .architectures
        .first()
        .cloned()
        .or_else(|| config.model_type.clone())
        .map(|name| {
            name.trim_end_matches("ForCausalLM")
                .trim_end_matches("ForConditionalGeneration")
                .to_ascii_lowercase()
        })
}

pub(crate) fn context_len(config: Option<&ConfigJson>, modalities: &[Modality]) -> usize {
    if modalities.contains(&Modality::Audio) {
        return config.and_then(|config| config.max_length).unwrap_or(448);
    }
    config
        .and_then(|config| {
            config
                .max_position_embeddings
                .or(config.max_source_positions)
        })
        .unwrap_or(2048)
}

pub(crate) fn detect_quantization(
    files: &[std::path::PathBuf],
    config: Option<&ConfigJson>,
) -> Option<Quantization> {
    let names = files
        .iter()
        .map(|path| path.to_string_lossy().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    for (needle, quantization) in [
        ("q4_k_m", Quantization::Q4KM),
        ("q5_k_m", Quantization::Q5KM),
        ("q6_k", Quantization::Q6K),
        ("q8_0", Quantization::Q8_0),
        ("q4_0", Quantization::Q4_0),
        ("int4", Quantization::Int4),
        ("int8", Quantization::Int8),
        ("fp32", Quantization::Fp32),
        ("fp16", Quantization::Fp16),
        ("bf16", Quantization::Bf16),
    ] {
        if names.contains(needle) {
            return Some(quantization);
        }
    }
    if let Some(bits) = config
        .and_then(|config| config.quantization_config.as_ref())
        .and_then(|config| config.bits)
    {
        return match bits {
            4 => Some(Quantization::Int4),
            8 => Some(Quantization::Int8),
            _ => None,
        };
    }
    config
        .and_then(|config| config.torch_dtype.as_deref())
        .and_then(|dtype| match dtype {
            "float32" => Some(Quantization::Fp32),
            "float16" => Some(Quantization::Fp16),
            "bfloat16" => Some(Quantization::Bf16),
            _ => None,
        })
}
