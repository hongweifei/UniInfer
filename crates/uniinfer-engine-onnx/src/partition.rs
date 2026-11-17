//! Split a single ONNX graph into sequential layer stages.
//!
//! ONNX Runtime exposes device placement per session, not per node. For a
//! sequential decoder we can still obtain true layer-level heterogeneity by
//! materializing one submodel per contiguous device group.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use onnx_protobuf::{GraphProto, Message, ModelProto, NodeProto, TensorProto, ValueInfoProto};
use protobuf::MessageField;
use uniinfer_core::{DeviceClass, EngineError, PlacementPlan, Result, TensorDtype, TensorSpec};

const RESERVE_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) struct Stage {
    pub(crate) name: String,
    pub(crate) device: DeviceClass,
    pub(crate) bytes: Vec<u8>,
    pub(crate) layers: usize,
}

pub(crate) struct Partitioned {
    pub(crate) stages: Vec<Stage>,
    pub(crate) inputs: Vec<TensorSpec>,
    pub(crate) outputs: Vec<TensorSpec>,
}

pub(crate) fn split(path: &Path, placement: &PlacementPlan) -> Result<Partitioned> {
    let raw = std::fs::read(path)
        .map_err(|error| EngineError::LoadFailed(format!("read {}: {error}", path.display())))?;
    let model = ModelProto::parse_from_bytes(&raw)
        .map_err(|error| EngineError::LoadFailed(format!("parse ONNX: {error}")))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("ONNX model has no graph".into()))?;
    let layer_count = graph
        .node
        .iter()
        .filter_map(|node| layer_index(&node.name))
        .max()
        .map_or(0, |layer| layer + 1);
    let has_layers = layer_count > 0;
    let (stage_devices, stage_layers, node_stage) = if has_layers {
        let first = graph
            .node
            .iter()
            .position(|node| layer_index(&node.name).is_some())
            .expect("layer count");
        let last = graph
            .node
            .iter()
            .rposition(|node| layer_index(&node.name).is_some())
            .expect("layer count");
        let gpu_layers = gpu_layers(path, layer_count, placement);
        let main_class = if gpu_layers > 0 {
            DeviceClass::Gpu
        } else {
            DeviceClass::Cpu
        };
        let mut groups: Vec<(DeviceClass, usize, usize)> = Vec::new();
        for layer in 0..layer_count {
            let device = if layer < gpu_layers {
                DeviceClass::Gpu
            } else {
                DeviceClass::Cpu
            };
            if let Some(last) = groups.last_mut()
                && last.0 == device
            {
                last.2 = layer + 1;
            } else {
                groups.push((device, layer, layer + 1));
            }
        }
        let stage_count = 2 + groups.len();
        let pre_stage = 0;
        let post_stage = stage_count - 1;
        let mut node_stage = vec![pre_stage; graph.node.len()];
        let mut current = pre_stage;
        for (index, node) in graph.node.iter().enumerate() {
            let stage = if let Some(layer) = layer_index(&node.name) {
                let group = groups
                    .iter()
                    .position(|(_, start, end)| (*start..*end).contains(&layer))
                    .unwrap_or(0);
                1 + group
            } else if index < first {
                pre_stage
            } else if index > last {
                post_stage
            } else {
                current
            };
            node_stage[index] = stage;
            current = stage;
        }
        let mut devices = vec![main_class];
        devices.extend(groups.iter().map(|(device, _, _)| *device));
        devices.push(main_class);
        let mut layers = vec![0];
        layers.extend(groups.iter().map(|(_, start, end)| end - start));
        layers.push(0);
        (devices, layers, node_stage)
    } else {
        let (split, devices) = generic_partition(graph, path, placement);
        let stage_count = devices.len();
        let node_stage = graph
            .node
            .iter()
            .enumerate()
            .map(|(index, _)| if split > 0 && index >= split { 1 } else { 0 })
            .map(|stage| stage.min(stage_count - 1))
            .collect::<Vec<_>>();
        (devices, vec![0; stage_count], node_stage)
    };
    let stage_count = stage_devices.len();

    let input_names = graph
        .input
        .iter()
        .map(|value| value.name.clone())
        .collect::<HashSet<_>>();
    let output_names = graph
        .output
        .iter()
        .map(|value| value.name.clone())
        .collect::<HashSet<_>>();
    let initializers = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<HashSet<_>>();
    let mut producer = HashMap::new();
    let mut producer_node = HashMap::new();
    for (index, node) in graph.node.iter().enumerate() {
        for output in &node.output {
            if !output.is_empty() {
                producer.insert(output.clone(), node_stage[index]);
                producer_node.insert(output.clone(), index);
            }
        }
    }
    let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, node) in graph.node.iter().enumerate() {
        for input in &node.input {
            if !input.is_empty() {
                consumers
                    .entry(input.clone())
                    .or_default()
                    .push(node_stage[index]);
            }
        }
    }

    let constant_nodes = graph
        .node
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op_type == "Constant")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let constant_outputs = constant_nodes
        .iter()
        .flat_map(|index| graph.node[*index].output.iter().cloned())
        .collect::<HashSet<_>>();
    let mut stage_inputs: Vec<Vec<String>> = vec![Vec::new(); stage_count];
    let mut stage_outputs: Vec<Vec<String>> = vec![Vec::new(); stage_count];
    let mut stage_initializers: Vec<Vec<String>> = vec![Vec::new(); stage_count];
    let mut stage_nodes: Vec<Vec<usize>> = vec![Vec::new(); stage_count];
    for (index, node) in graph.node.iter().enumerate() {
        let stage = node_stage[index];
        stage_nodes[stage].push(index);
        for input in &node.input {
            if input.is_empty() || constant_outputs.contains(input) {
                continue;
            }
            if input_names.contains(input) {
                push_unique(&mut stage_inputs[stage], input);
            } else if initializers.contains(input) {
                push_unique(&mut stage_initializers[stage], input);
            } else if producer.get(input) == Some(&stage) {
                // Produced and consumed inside this stage.
            } else if producer
                .get(input)
                .is_some_and(|producer| *producer < stage)
            {
                push_unique(&mut stage_inputs[stage], input);
            } else {
                return Err(EngineError::LoadFailed(format!(
                    "layer partition cannot resolve input {input} for stage {stage}"
                )));
            }
        }
        for output in &node.output {
            if output.is_empty() {
                continue;
            }
            let used_later = consumers
                .get(output)
                .is_some_and(|stages| stages.iter().any(|consumer| *consumer > stage));
            if output_names.contains(output) || used_later {
                push_unique(&mut stage_outputs[stage], output);
            }
        }
    }

    let value_info = value_info_map(graph);
    let input_info = graph
        .input
        .iter()
        .map(|value| (value.name.clone(), value.clone()))
        .collect::<HashMap<_, _>>();
    let output_info = graph
        .output
        .iter()
        .map(|value| (value.name.clone(), value.clone()))
        .collect::<HashMap<_, _>>();
    let initializers = graph
        .initializer
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.clone()))
        .collect::<HashMap<_, _>>();

    for nodes in &mut stage_nodes {
        nodes.extend(constant_nodes.iter().copied());
        nodes.sort_unstable();
        nodes.dedup();
    }
    let mut stages = Vec::with_capacity(stage_count);
    for stage in 0..stage_count {
        let inputs = stage_inputs[stage]
            .iter()
            .map(|name| {
                input_info
                    .get(name)
                    .or_else(|| value_info.get(name))
                    .cloned()
                    .ok_or_else(|| {
                        EngineError::LoadFailed(format!("missing ONNX type for stage input {name}"))
                    })
            })
            .collect::<Result<Vec<ValueInfoProto>>>()?;
        let outputs = stage_outputs[stage]
            .iter()
            .map(|name| {
                output_info
                    .get(name)
                    .or_else(|| value_info.get(name))
                    .cloned()
                    .ok_or_else(|| {
                        EngineError::LoadFailed(format!(
                            "missing ONNX type for stage output {name}"
                        ))
                    })
            })
            .collect::<Result<Vec<ValueInfoProto>>>()?;
        let initializers = stage_initializers[stage]
            .iter()
            .filter_map(|name| initializers.get(name).cloned())
            .collect::<Vec<TensorProto>>();
        let nodes = stage_nodes[stage]
            .iter()
            .map(|index| graph.node[*index].clone())
            .collect::<Vec<NodeProto>>();
        let device = stage_devices[stage];
        let graph = GraphProto {
            node: nodes,
            name: format!("uniinfer-stage-{stage}"),
            initializer: initializers,
            input: inputs,
            output: outputs,
            ..GraphProto::default()
        };
        let submodel = ModelProto {
            ir_version: model.ir_version,
            opset_import: model.opset_import.clone(),
            producer_name: "uniinfer".into(),
            graph: MessageField::some(graph),
            ..ModelProto::default()
        };
        let bytes = submodel
            .write_to_bytes()
            .map_err(|error| EngineError::LoadFailed(format!("serialize ONNX stage: {error}")))?;
        let layers = stage_layers[stage];
        stages.push(Stage {
            name: format!("stage-{stage}"),
            device,
            bytes,
            layers,
        });
    }
    Ok(Partitioned {
        stages,
        inputs: value_specs(&graph.input)?,
        outputs: value_specs(&graph.output)?,
    })
}

fn gpu_layers(path: &Path, layer_count: usize, placement: &PlacementPlan) -> usize {
    if placement
        .devices
        .first()
        .is_none_or(|device| device.class != DeviceClass::Gpu)
    {
        return 0;
    }
    if !placement.auto_offload {
        return placement.offload_layers.min(layer_count);
    }
    let available = placement
        .devices
        .iter()
        .find(|device| device.class == DeviceClass::Gpu)
        .map_or(0, |device| device.available_memory_bytes);
    let model_bytes = std::fs::metadata(path).map_or(0, |metadata| metadata.len());
    if available == 0 || model_bytes == 0 {
        return 0;
    }
    let per_layer = model_bytes.div_ceil(layer_count as u64);
    let usable = available.saturating_sub(RESERVE_BYTES);
    usize::try_from(usable / per_layer)
        .unwrap_or(usize::MAX)
        .min(layer_count)
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|known| known == value) {
        values.push(value.to_owned());
    }
}

fn layer_index(name: &str) -> Option<usize> {
    let marker = name.find("/layers.").or_else(|| name.find("layers."))?;
    let rest = &name[marker + "/layers.".len()..];
    let digits = rest
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().ok()
}

fn value_info_map(graph: &GraphProto) -> HashMap<String, ValueInfoProto> {
    graph
        .value_info
        .iter()
        .map(|value| (value.name.clone(), value.clone()))
        .collect()
}

fn generic_partition(
    graph: &GraphProto,
    _path: &Path,
    placement: &PlacementPlan,
) -> (usize, Vec<DeviceClass>) {
    if placement
        .devices
        .first()
        .is_none_or(|device| device.class != DeviceClass::Gpu)
    {
        return (0, vec![DeviceClass::Cpu]);
    }
    let total = graph
        .initializer
        .iter()
        .map(|tensor| u64::try_from(tensor.raw_data.len()).unwrap_or(0))
        .sum::<u64>();
    let available = placement
        .devices
        .iter()
        .find(|device| device.class == DeviceClass::Gpu)
        .map_or(0, |device| device.available_memory_bytes);
    if available == 0 || total == 0 {
        return (0, vec![DeviceClass::Gpu]);
    }
    if available >= total.saturating_add(RESERVE_BYTES) {
        return (0, vec![DeviceClass::Gpu]);
    }
    let budget = available.saturating_sub(RESERVE_BYTES);
    if budget == 0 {
        return (0, vec![DeviceClass::Cpu]);
    }
    let initializer = graph
        .initializer
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                u64::try_from(tensor.raw_data.len()).unwrap_or(0),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut cumulative = 0u64;
    for (index, node) in graph.node.iter().enumerate() {
        cumulative = cumulative.saturating_add(
            node.input
                .iter()
                .filter_map(|name| initializer.get(name))
                .copied()
                .sum::<u64>(),
        );
        if cumulative >= budget && index + 1 < graph.node.len() {
            return (index + 1, vec![DeviceClass::Gpu, DeviceClass::Cpu]);
        }
    }
    (0, vec![DeviceClass::Gpu])
}

fn value_specs(values: &[ValueInfoProto]) -> Result<Vec<TensorSpec>> {
    values
        .iter()
        .map(|value| {
            let ty = value
                .type_
                .as_ref()
                .and_then(|ty| match &ty.value {
                    Some(onnx_protobuf::type_proto::Value::TensorType(tensor)) => Some(tensor),
                    _ => None,
                })
                .ok_or_else(|| {
                    EngineError::LoadFailed(format!("ONNX value {} is not a tensor", value.name))
                })?;
            let dtype = match ty.elem_type {
                1 => TensorDtype::F32,
                10 => TensorDtype::F16,
                6 => TensorDtype::I32,
                7 => TensorDtype::I64,
                2 => TensorDtype::U8,
                9 => TensorDtype::Bool,
                other => {
                    return Err(EngineError::LoadFailed(format!(
                        "ONNX value {} has unsupported dtype {other}",
                        value.name
                    )));
                }
            };
            let shape = ty
                .shape
                .as_ref()
                .map(|shape| {
                    shape
                        .dim
                        .iter()
                        .map(|dim| match dim.value {
                            Some(
                                onnx_protobuf::tensor_shape_proto::dimension::Value::DimValue(
                                    value,
                                ),
                            ) => usize::try_from(value).ok(),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(TensorSpec {
                name: value.name.clone(),
                dtype,
                shape,
            })
        })
        .collect()
}
