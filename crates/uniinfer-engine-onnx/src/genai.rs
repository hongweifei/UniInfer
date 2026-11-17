//! ONNX Runtime GenAI metadata and KV-cache generation.
//!
//! Cache values are ordinary `Tensor`s (F32 or F16), so GenAI uses the same
//! `Graph` execution, output conversion and cancellation path as other ONNX
//! sessions.

use std::path::Path;

use serde::Deserialize;
use uniinfer_core::{EngineError, Result, Tensor, TensorData, TensorDtype, TensorSpec};

use crate::graph::Graph;

#[derive(Debug, Deserialize)]
pub(crate) struct GenaiConfig {
    pub(crate) model: GenaiModel,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenaiModel {
    pub(crate) decoder: GenaiDecoder,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenaiDecoder {
    pub(crate) inputs: GenaiInputs,
    pub(crate) outputs: GenaiOutputs,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_size: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenaiInputs {
    pub(crate) input_ids: String,
    pub(crate) attention_mask: String,
    pub(crate) past_key_names: String,
    pub(crate) past_value_names: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenaiOutputs {
    pub(crate) logits: String,
    pub(crate) present_key_names: String,
    pub(crate) present_value_names: String,
}

#[derive(Debug, Clone)]
pub(crate) struct GenaiSpec {
    input_ids: String,
    attention_mask: String,
    logits: String,
    past_order: Vec<String>,
    present_order: Vec<String>,
    kv_heads: usize,
    head_size: usize,
    past_dtype: TensorDtype,
}

impl GenaiSpec {
    pub(crate) fn from_config(config: &GenaiConfig, graph: &Graph) -> Result<Self> {
        let decoder = &config.model.decoder;
        if decoder.num_hidden_layers == 0
            || decoder.num_key_value_heads == 0
            || decoder.head_size == 0
        {
            return Err(EngineError::ManifestInvalid(
                "genai_config has zero layers/heads/head_size".into(),
            ));
        }
        let names = |pattern: &str| {
            (0..decoder.num_hidden_layers)
                .map(|layer| pattern.replace("%d", &layer.to_string()))
                .collect::<Vec<_>>()
        };
        let past_keys = names(&decoder.inputs.past_key_names);
        let past_values = names(&decoder.inputs.past_value_names);
        let present_keys = names(&decoder.outputs.present_key_names);
        let present_values = names(&decoder.outputs.present_value_names);
        let past_dtype = graph
            .inputs()
            .iter()
            .find(|spec| spec.name == past_keys[0])
            .map(|spec| spec.dtype)
            .ok_or_else(|| {
                EngineError::ManifestInvalid(format!(
                    "genai past input {} is missing",
                    past_keys[0]
                ))
            })?;
        if !matches!(past_dtype, TensorDtype::F32 | TensorDtype::F16) {
            return Err(EngineError::LoadFailed(format!(
                "unsupported genai KV dtype {past_dtype:?}"
            )));
        }
        let past_order = graph
            .inputs()
            .iter()
            .map(|spec| spec.name.clone())
            .filter(|name| past_keys.contains(name) || past_values.contains(name))
            .collect::<Vec<_>>();
        let present_order = graph
            .outputs()
            .iter()
            .map(|spec| spec.name.clone())
            .filter(|name| present_keys.contains(name) || present_values.contains(name))
            .collect::<Vec<_>>();
        if past_order.len() != decoder.num_hidden_layers * 2
            || present_order.len() != decoder.num_hidden_layers * 2
        {
            return Err(EngineError::ManifestInvalid(
                "genai KV input/output names do not match layer count".into(),
            ));
        }
        Ok(Self {
            input_ids: decoder.inputs.input_ids.clone(),
            attention_mask: decoder.inputs.attention_mask.clone(),
            logits: decoder.outputs.logits.clone(),
            past_order,
            present_order,
            kv_heads: decoder.num_key_value_heads,
            head_size: decoder.head_size,
            past_dtype,
        })
    }

    pub(crate) fn input_specs(&self, graph: &Graph) -> Result<Vec<TensorSpec>> {
        let wanted = [&self.input_ids, &self.attention_mask];
        let filtered = graph
            .inputs()
            .iter()
            .filter(|spec| wanted.contains(&&spec.name))
            .cloned()
            .collect::<Vec<_>>();
        if filtered.len() != 2 {
            return Err(EngineError::ManifestInvalid(
                "genai input_ids/attention_mask are missing".into(),
            ));
        }
        Ok(filtered)
    }

    pub(crate) fn output_specs(&self, graph: &Graph) -> Result<Vec<TensorSpec>> {
        let filtered = graph
            .outputs()
            .iter()
            .filter(|spec| spec.name == self.logits)
            .cloned()
            .collect::<Vec<_>>();
        if filtered.len() != 1 {
            return Err(EngineError::ManifestInvalid(
                "genai logits output is missing".into(),
            ));
        }
        Ok(filtered)
    }
}

pub(crate) struct GenaiState {
    spec: GenaiSpec,
    past: Vec<Tensor>,
    cached_ids: Vec<i64>,
}

impl GenaiState {
    pub(crate) fn new(spec: GenaiSpec) -> Self {
        Self {
            spec,
            past: Vec::new(),
            cached_ids: Vec::new(),
        }
    }

    pub(crate) fn forward(&mut self, graph: &mut Graph, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == self.spec.input_ids)
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("genai input_ids missing".into()))?;
        if ids.is_empty() {
            return Err(EngineError::Other("genai input_ids is empty".into()));
        }
        let mask = inputs
            .iter()
            .find(|tensor| tensor.name == self.spec.attention_mask)
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) if values.len() == ids.len() => Some(values.clone()),
                _ => None,
            })
            .unwrap_or_else(|| vec![1; ids.len()]);

        let reuse = !self.cached_ids.is_empty()
            && ids.len() > self.cached_ids.len()
            && ids.starts_with(&self.cached_ids);
        let mut past = if reuse {
            std::mem::take(&mut self.past)
        } else {
            zero_past(&self.spec)?
        };
        if past.is_empty() {
            past = zero_past(&self.spec)?;
        }
        let delta_start = if reuse { self.cached_ids.len() } else { 0 };
        let delta = &ids[delta_start..];
        if delta.is_empty() {
            return Err(EngineError::Other("genai input has no new tokens".into()));
        }

        let mut values = Vec::with_capacity(2 + past.len());
        values.push(Tensor::new(
            self.spec.input_ids.clone(),
            vec![1, delta.len()],
            TensorData::I64(delta.to_vec()),
        )?);
        values.push(Tensor::new(
            self.spec.attention_mask.clone(),
            vec![1, mask.len()],
            TensorData::I64(mask),
        )?);
        values.extend(past);

        let outputs = graph.run(values)?;
        let logits = outputs
            .iter()
            .find(|tensor| tensor.name == self.spec.logits)
            .cloned()
            .ok_or_else(|| EngineError::Other("genai logits missing".into()))?;
        let presents = self
            .spec
            .past_order
            .iter()
            .zip(self.spec.present_order.iter())
            .map(|(past, present)| {
                let tensor = outputs
                    .iter()
                    .find(|tensor| tensor.name == *present)
                    .cloned()
                    .ok_or_else(|| EngineError::Other(format!("genai output {present} missing")))?;
                Tensor::new(past.clone(), tensor.shape, tensor.data)
            })
            .collect::<Result<Vec<_>>>()?;
        self.past = presents;
        self.cached_ids = ids;
        Ok(vec![logits])
    }
}

pub(crate) fn load_genai_config(base_dir: &Path) -> Result<Option<GenaiConfig>> {
    let path = base_dir.join("genai_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path).map_err(|error| {
        EngineError::ManifestInvalid(format!("cannot read {}: {error}", path.display()))
    })?;
    let config = serde_json::from_str(&raw)
        .map_err(|error| EngineError::ManifestInvalid(format!("{}: {error}", path.display())))?;
    Ok(Some(config))
}

fn zero_past(spec: &GenaiSpec) -> Result<Vec<Tensor>> {
    let shape = vec![1usize, spec.kv_heads, 0, spec.head_size];
    let data = match spec.past_dtype {
        TensorDtype::F32 => TensorData::F32(Vec::new()),
        TensorDtype::F16 => TensorData::F16(Vec::new()),
        other => {
            return Err(EngineError::LoadFailed(format!(
                "unsupported genai KV dtype {other:?}"
            )));
        }
    };
    spec.past_order
        .iter()
        .map(|name| Tensor::new(name.clone(), shape.clone(), data.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_genai_config_from_model_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("genai_config.json"),
            r#"{
                "model": {
                    "vocab_size": 151936,
                    "eos_token_id": 151645,
                    "decoder": {
                        "inputs": {
                            "input_ids": "input_ids",
                            "attention_mask": "attention_mask",
                            "past_key_names": "past_key_values.%d.key",
                            "past_value_names": "past_key_values.%d.value"
                        },
                        "outputs": {
                            "logits": "logits",
                            "present_key_names": "present.%d.key",
                            "present_value_names": "present.%d.value"
                        },
                        "num_hidden_layers": 28,
                        "num_key_value_heads": 8,
                        "head_size": 128
                    }
                }
            }"#,
        )
        .expect("write config");
        let config = load_genai_config(dir.path())
            .expect("parse config")
            .expect("config present");
        assert_eq!(config.model.decoder.num_hidden_layers, 28);
        assert_eq!(config.model.decoder.num_key_value_heads, 8);
        assert_eq!(config.model.decoder.head_size, 128);
    }
}
