//! Decoder execution and KV-cache hand-off for Whisper ONNX.

use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor as OrtTensor};
use uniinfer_core::{EngineError, Result};

use crate::asr::decode::argmax;
pub(super) fn run_decoder(
    session: &mut Session,
    ids: &[u32],
    hidden_shape: &[usize],
    hidden_values: &[f32],
) -> Result<(u32, Vec<(String, DynValue)>)> {
    let input_ids_name = session
        .inputs()
        .first()
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no inputs".into()))?;
    let hidden_name = session
        .inputs()
        .get(1)
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no encoder input".into()))?;
    let logits_name = session
        .outputs()
        .first()
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no outputs".into()))?;
    let input_ids = OrtTensor::from_array((
        vec![1usize, ids.len()],
        ids.iter().map(|id| i64::from(*id)).collect::<Vec<_>>(),
    ))
    .map_err(|error| EngineError::Other(format!("decoder ids: {error}")))?
    .into_dyn();
    let hidden = OrtTensor::from_array((hidden_shape.to_vec(), hidden_values.to_vec()))
        .map_err(|error| EngineError::Other(format!("encoder hidden: {error}")))?
        .into_dyn();
    let present_names = present_names(session);
    let mut outputs = session
        .run(vec![(input_ids_name, input_ids), (hidden_name, hidden)])
        .map_err(|error| EngineError::Other(format!("onnx decoder: {error}")))?;
    let token = token_from_logits(&outputs, &logits_name)?;
    let cache = take_present(&mut outputs, &present_names)?;
    Ok((token, cache))
}

pub(super) fn run_with_past(
    session: &mut Session,
    token: u32,
    cache: Vec<(String, DynValue)>,
    position: usize,
) -> Result<(u32, Vec<(String, DynValue)>)> {
    let input_ids_name = session
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == "input_ids")
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder_with_past has no input_ids".into()))?;
    let logits_name = session
        .outputs()
        .first()
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder_with_past has no outputs".into()))?;
    let input_names = session
        .inputs()
        .iter()
        .map(|outlet| outlet.name().to_owned())
        .collect::<Vec<_>>();

    let ids_value = OrtTensor::from_array((vec![1usize, 1usize], vec![i64::from(token)]))
        .map_err(|error| EngineError::Other(format!("past ids: {error}")))?
        .into_dyn();
    let position_value = OrtTensor::from_array((vec![1usize], vec![position as i64]))
        .map_err(|error| EngineError::Other(format!("cache position: {error}")))?
        .into_dyn();
    let mut values: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(input_names.len());
    for name in input_names {
        match name.as_str() {
            "input_ids" => {
                values.push((input_ids_name.clone(), SessionInputValue::from(&ids_value)));
            }
            "cache_position" => {
                values.push((name, SessionInputValue::from(&position_value)));
            }
            _ => {
                let present = name.replacen("past_key_values.", "present.", 1);
                let value = cache
                    .iter()
                    .find(|(cached, _)| cached == &present)
                    .map(|(_, value)| value)
                    .ok_or_else(|| EngineError::Other(format!("missing past cache {present}")))?;
                values.push((name, SessionInputValue::from(value)));
            }
        }
    }
    let present_names = present_names(session);
    let mut outputs = session
        .run(values)
        .map_err(|error| EngineError::Other(format!("onnx decoder_with_past: {error}")))?;
    let token = token_from_logits(&outputs, &logits_name)?;
    let retained = cache
        .into_iter()
        .filter(|(name, _)| name.contains(".encoder."))
        .collect::<Vec<_>>();
    let mut cache = take_present(&mut outputs, &present_names)?;
    cache.extend(retained);
    Ok((token, cache))
}

fn token_from_logits(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<u32> {
    let (shape, logits) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|error| EngineError::Other(format!("decoder logits: {error}")))?;
    let shape = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    let vocab = *shape
        .last()
        .ok_or_else(|| EngineError::Other("empty logits shape".into()))?;
    let offset = logits.len().saturating_sub(vocab);
    Ok(argmax(&logits[offset..]))
}

fn present_names(session: &Session) -> Vec<String> {
    session
        .outputs()
        .iter()
        .map(|outlet| outlet.name().to_owned())
        .filter(|name| name.starts_with("present."))
        .collect()
}

fn take_present(
    outputs: &mut ort::session::SessionOutputs<'_>,
    names: &[String],
) -> Result<Vec<(String, DynValue)>> {
    let mut values = Vec::with_capacity(names.len());
    for name in names {
        let value = outputs
            .remove(name.as_str())
            .ok_or_else(|| EngineError::Other(format!("missing present cache {name}")))?;
        values.push((name.clone(), value));
    }
    Ok(values)
}
