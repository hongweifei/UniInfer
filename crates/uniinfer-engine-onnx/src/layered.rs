//! Layer-partitioned ONNX decoder.
//!
//! Each contiguous device group becomes an independent ORT session. The
//! session keeps KV tensors between calls and feeds inter-stage hidden states.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, Residency, Result, Tensor, TensorData,
    TensorDtype, TensorSpec,
};

use crate::genai::load_genai_config;
use crate::graph::Graph;
use crate::partition;

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    _class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let config = load_genai_config(&spec.base_dir)?.ok_or_else(|| {
        EngineError::ManifestInvalid("onnx.layered requires genai_config.json".into())
    })?;
    let model_file = spec
        .manifest
        .files
        .first()
        .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
    let model_path = spec.base_dir.join(&model_file.path);
    let stages = partition::split(&model_path, &spec.placement)?;
    let decoder = &config.model.decoder;
    let past = (0..decoder.num_hidden_layers)
        .flat_map(|layer| {
            [
                (
                    decoder
                        .inputs
                        .past_key_names
                        .replace("%d", &layer.to_string()),
                    decoder
                        .outputs
                        .present_key_names
                        .replace("%d", &layer.to_string()),
                ),
                (
                    decoder
                        .inputs
                        .past_value_names
                        .replace("%d", &layer.to_string()),
                    decoder
                        .outputs
                        .present_value_names
                        .replace("%d", &layer.to_string()),
                ),
            ]
        })
        .collect::<Vec<_>>();
    let mut graphs = Vec::with_capacity(stages.len());
    let mut vram_bytes = 0u64;
    let mut host_bytes = 0u64;
    let mut offload_layers = 0usize;
    let disk_bytes = std::fs::metadata(&model_path).map_or(0, |metadata| metadata.len());
    for stage in stages {
        let bytes = u64::try_from(stage.bytes.len()).unwrap_or(0);
        if stage.device == DeviceClass::Gpu {
            vram_bytes = vram_bytes.saturating_add(bytes);
            offload_layers += stage.layers;
        } else {
            host_bytes = host_bytes.saturating_add(bytes);
        }
        graphs.push(Mutex::new(Graph::from_bytes(
            environment,
            stage.device,
            &stage.bytes,
            &stage.name,
        )?));
    }
    let residency = Residency {
        vram_bytes,
        host_bytes,
        disk_bytes,
        offload_layers,
        total_layers: Some(decoder.num_hidden_layers),
        estimated: false,
    };
    let input_ids = decoder.inputs.input_ids.clone();
    let attention_mask = decoder.inputs.attention_mask.clone();
    let logits = decoder.outputs.logits.clone();
    Ok(Box::new(LayeredSession {
        graphs,
        input_ids: input_ids.clone(),
        attention_mask: attention_mask.clone(),
        logits: logits.clone(),
        past,
        state: Mutex::new(LayeredState {
            past_values: HashMap::new(),
            cached_ids: Vec::new(),
        }),
        inputs: vec![
            TensorSpec {
                name: input_ids,
                dtype: TensorDtype::I64,
                shape: vec![Some(1), None],
            },
            TensorSpec {
                name: attention_mask,
                dtype: TensorDtype::I64,
                shape: vec![Some(1), None],
            },
        ],
        outputs: vec![TensorSpec {
            name: logits.clone(),
            dtype: TensorDtype::F32,
            shape: vec![Some(1), None, None],
        }],
        residency,
        cancelled: AtomicBool::new(false),
    }))
}

struct LayeredSession {
    graphs: Vec<Mutex<Graph>>,
    input_ids: String,
    attention_mask: String,
    logits: String,
    past: Vec<(String, String)>,
    state: Mutex<LayeredState>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    residency: Residency,
    cancelled: AtomicBool,
}

struct LayeredState {
    past_values: HashMap<String, Tensor>,
    cached_ids: Vec<i64>,
}

impl EngineSession for LayeredSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn residency(&self) -> Option<Residency> {
        Some(self.residency)
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let ids = find_i64(&inputs, &self.input_ids)?;
        let mask = find_i64(&inputs, &self.attention_mask).unwrap_or_else(|_| vec![1; ids.len()]);
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("layered state lock poisoned".into()))?;
        let reuse = !state.cached_ids.is_empty()
            && ids.len() > state.cached_ids.len()
            && ids.starts_with(&state.cached_ids);
        let delta = if reuse {
            &ids[state.cached_ids.len()..]
        } else {
            ids.as_slice()
        };
        let mut values = HashMap::new();
        values.insert(
            self.input_ids.clone(),
            Tensor::new(
                self.input_ids.clone(),
                vec![1, delta.len()],
                TensorData::I64(delta.to_vec()),
            )?,
        );
        values.insert(
            self.attention_mask.clone(),
            Tensor::new(
                self.attention_mask.clone(),
                vec![1, mask.len()],
                TensorData::I64(mask.clone()),
            )?,
        );
        let mut past_values = if reuse {
            std::mem::take(&mut state.past_values)
        } else {
            HashMap::new()
        };
        let mut logits = None;
        for graph in &self.graphs {
            let mut graph = graph
                .lock()
                .map_err(|_| EngineError::Other("layer partition lock poisoned".into()))?;
            let mut stage_inputs = Vec::with_capacity(graph.inputs().len());
            for spec in graph.inputs() {
                let value = match values
                    .remove(&spec.name)
                    .or_else(|| past_values.remove(&spec.name))
                {
                    Some(value) => value,
                    None => zero_tensor(spec)?,
                };
                stage_inputs.push(value);
            }
            let outputs = graph.run(stage_inputs)?;
            for output in outputs {
                if output.name == self.logits {
                    logits = Some(output.clone());
                }
                values.insert(output.name.clone(), output);
            }
            for (past_name, present_name) in &self.past {
                if let Some(mut value) = values.get(present_name).cloned() {
                    value.name = past_name.clone();
                    past_values.insert(past_name.clone(), value);
                }
            }
        }
        state.past_values = past_values;
        state.cached_ids = ids;
        logits
            .map(|logits| vec![logits])
            .ok_or_else(|| EngineError::Other("layered ONNX did not emit logits".into()))
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

fn find_i64(inputs: &[Tensor], name: &str) -> Result<Vec<i64>> {
    inputs
        .iter()
        .find(|tensor| tensor.name == name)
        .and_then(|tensor| match &tensor.data {
            TensorData::I64(values) => Some(values.clone()),
            _ => None,
        })
        .ok_or_else(|| EngineError::Other(format!("missing int64 input {name}")))
}

fn zero_tensor(spec: &TensorSpec) -> Result<Tensor> {
    let rank = spec.shape.len();
    let shape = spec
        .shape
        .iter()
        .enumerate()
        .map(|(index, dim)| {
            dim.unwrap_or(if index == 0 {
                1
            } else if rank >= 3 && index + 2 == rank {
                0
            } else {
                1
            })
        })
        .collect::<Vec<_>>();
    let data = match spec.dtype {
        TensorDtype::F32 => TensorData::F32(Vec::new()),
        TensorDtype::F16 => TensorData::F16(Vec::new()),
        other => {
            return Err(EngineError::LoadFailed(format!(
                "unsupported past dtype {other:?}"
            )));
        }
    };
    Tensor::new(spec.name.clone(), shape, data)
}
