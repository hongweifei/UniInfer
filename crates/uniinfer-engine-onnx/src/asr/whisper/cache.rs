//! Decoder execution and KV-cache hand-off for Whisper ONNX.
//!
//! This module only deals in `Tensor` and `Graph`; ORT values stay inside the
//! graph wrapper.

use uniinfer_core::{EngineError, Result, Tensor, TensorData};

use crate::asr::decode::argmax;
use crate::graph::Graph;

pub(super) fn run_decoder(
    graph: &mut Graph,
    ids: &[u32],
    encoder_hidden: &Tensor,
) -> Result<(u32, Vec<Tensor>)> {
    let input_ids_name = graph
        .inputs()
        .iter()
        .find(|spec| spec.name == "input_ids")
        .map(|spec| spec.name.clone())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no input_ids".into()))?;
    let hidden_name = graph
        .inputs()
        .iter()
        .find(|spec| spec.name != input_ids_name)
        .map(|spec| spec.name.clone())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no encoder input".into()))?;
    let input_ids = Tensor::new(
        input_ids_name,
        vec![1, ids.len()],
        TensorData::I64(ids.iter().map(|id| i64::from(*id)).collect()),
    )?;
    let hidden = Tensor::new(
        hidden_name,
        encoder_hidden.shape.clone(),
        encoder_hidden.data.clone(),
    )?;
    let outputs = graph.run(vec![input_ids, hidden])?;
    let token = token_from_logits(&outputs)?;
    let cache = take_present(graph, &outputs);
    Ok((token, cache))
}

pub(super) fn run_with_past(
    graph: &mut Graph,
    token: u32,
    cache: Vec<Tensor>,
    position: usize,
) -> Result<(u32, Vec<Tensor>)> {
    let specs = graph.inputs().to_vec();
    let mut inputs = Vec::with_capacity(specs.len());
    for spec in specs {
        let tensor = match spec.name.as_str() {
            "input_ids" => Tensor::new(
                spec.name,
                vec![1, 1],
                TensorData::I64(vec![i64::from(token)]),
            )?,
            "cache_position" => {
                Tensor::new(spec.name, vec![1], TensorData::I64(vec![position as i64]))?
            }
            name => {
                let present = name.replacen("past_key_values.", "present.", 1);
                let tensor = cache
                    .iter()
                    .find(|tensor| tensor.name == present)
                    .cloned()
                    .ok_or_else(|| EngineError::Other(format!("missing past cache {present}")))?;
                Tensor::new(spec.name, tensor.shape, tensor.data)?
            }
        };
        inputs.push(tensor);
    }

    let outputs = graph.run(inputs)?;
    let token = token_from_logits(&outputs)?;
    let retained = cache
        .into_iter()
        .filter(|tensor| tensor.name.contains(".encoder."))
        .collect::<Vec<_>>();
    let mut cache = take_present(graph, &outputs);
    cache.extend(retained);
    Ok((token, cache))
}

fn token_from_logits(outputs: &[Tensor]) -> Result<u32> {
    let logits = outputs
        .iter()
        .find(|tensor| tensor.name == "logits")
        .or_else(|| outputs.first())
        .ok_or_else(|| EngineError::Other("decoder has no logits".into()))?;
    let TensorData::F32(values) = &logits.data else {
        return Err(EngineError::Other("decoder logits must be f32".into()));
    };
    let vocab = *logits
        .shape
        .last()
        .ok_or_else(|| EngineError::Other("empty logits shape".into()))?;
    let offset = values.len().saturating_sub(vocab);
    Ok(argmax(&values[offset..]))
}

fn take_present(graph: &Graph, outputs: &[Tensor]) -> Vec<Tensor> {
    graph
        .outputs()
        .iter()
        .filter(|spec| spec.name.starts_with("present."))
        .filter_map(|spec| {
            outputs
                .iter()
                .find(|tensor| tensor.name == spec.name)
                .cloned()
        })
        .collect()
}
