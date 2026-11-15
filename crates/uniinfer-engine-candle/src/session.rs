//! Candle session: token bookkeeping, KV cache and logits extraction.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use uniinfer_core::{
    EngineError, EngineSession, Result, Tensor as CoreTensor, TensorData, TensorDtype, TensorSpec,
};

use crate::text::TextModel;

pub(crate) struct CandleSession {
    state: Mutex<CandleState>,
    cancelled: AtomicBool,
    device: Device,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

struct CandleState {
    model: Box<dyn TextModel>,
    cached_ids: Vec<u32>,
}

impl CandleSession {
    pub(crate) fn new(model: Box<dyn TextModel>, device: Device, vocab: usize) -> Self {
        Self {
            state: Mutex::new(CandleState {
                model,
                cached_ids: Vec::new(),
            }),
            cancelled: AtomicBool::new(false),
            device,
            inputs: vec![TensorSpec {
                name: "input_ids".into(),
                dtype: TensorDtype::I64,
                shape: vec![Some(1), None],
            }],
            outputs: vec![TensorSpec {
                name: "logits".into(),
                dtype: TensorDtype::F32,
                shape: vec![Some(1), None, Some(vocab)],
            }],
        }
    }
}

impl EngineSession for CandleSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<CoreTensor>) -> Result<Vec<CoreTensor>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == "input_ids")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("missing int64 input_ids".into()))?;
        let ids = ids
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| EngineError::Other("id overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("candle state lock poisoned".into()))?;
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let incremental = state.cached_ids.len() + 1 == ids.len()
            && ids.starts_with(&state.cached_ids)
            && !state.cached_ids.is_empty();
        let (tokens, offset) = if incremental {
            (&ids[ids.len() - 1..], state.cached_ids.len())
        } else {
            state.model.clear();
            state.cached_ids.clear();
            (ids.as_slice(), 0)
        };
        let input = Tensor::new(tokens, &self.device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| EngineError::Other(format!("input tensor: {error}")))?;
        let logits = state
            .model
            .forward(&input, offset)
            .map_err(|error| EngineError::Other(format!("candle forward: {error}")))?;
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|error| EngineError::Other(format!("logits dtype: {error}")))?;
        let shape = logits.dims().to_vec();
        let values = logits
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| EngineError::Other(format!("logits: {error}")))?;
        state.cached_ids = ids;
        Ok(vec![CoreTensor::new(
            "logits",
            shape,
            TensorData::F32(values),
        )?])
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}
