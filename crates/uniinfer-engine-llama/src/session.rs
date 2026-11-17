//! llama.cpp session state, KV cache and decode loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use self_cell::self_cell;
use uniinfer_core::{
    EngineError, EngineSession, Residency, Result, Tensor as CoreTensor, TensorData, TensorSpec,
};

use crate::BATCH_TOKENS;

self_cell!(
    struct ModelContext {
        owner: Arc<LlamaModel>,
        #[covariant]
        dependent: LlamaContext,
    }
);

struct LlamaState {
    context: ModelContext,
    cached_ids: Vec<u32>,
}

pub(crate) struct LlamaSession {
    state: Mutex<LlamaState>,
    cancelled: AtomicBool,
    residency: Residency,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl EngineSession for LlamaSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn residency(&self) -> Option<Residency> {
        Some(self.residency)
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
        if ids.is_empty() {
            return Err(EngineError::Other("input_ids is empty".into()));
        }
        let ids = ids
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| EngineError::Other("id overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("llama state lock poisoned".into()))?;
        let incremental = state.cached_ids.len() + 1 == ids.len()
            && ids.starts_with(&state.cached_ids)
            && !state.cached_ids.is_empty();
        let (tokens, offset) = if incremental {
            (&ids[ids.len() - 1..], state.cached_ids.len())
        } else {
            state.cached_ids.clear();
            state
                .context
                .with_dependent_mut(|_, context| context.clear_kv_cache());
            (ids.as_slice(), 0)
        };
        let logits = state.context.with_dependent_mut(|_, context| {
            decode_chunked(
                context,
                tokens,
                offset,
                usize::try_from(context.n_batch()).unwrap_or(BATCH_TOKENS),
            )
        })?;
        state.cached_ids = ids;
        let vocab = logits.len();
        Ok(vec![CoreTensor::new(
            "logits",
            vec![1, 1, vocab],
            TensorData::F32(logits),
        )?])
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

fn decode_chunked(
    context: &mut LlamaContext<'_>,
    tokens: &[u32],
    offset: usize,
    chunk: usize,
) -> Result<Vec<f32>> {
    let chunk = chunk.max(1);
    let last = tokens.len().saturating_sub(1);
    for (start, window) in tokens.chunks(chunk).enumerate() {
        let base = offset + start * chunk;
        let mut batch = LlamaBatch::new(window.len(), 1);
        for (index, token) in window.iter().enumerate() {
            let token = i32::try_from(*token)
                .map_err(|_| EngineError::Other("token id overflow".into()))?;
            let position = i32::try_from(base + index)
                .map_err(|_| EngineError::Other("position overflow".into()))?;
            batch
                .add(
                    LlamaToken(token),
                    position,
                    &[0],
                    base + index == offset + last,
                )
                .map_err(|error| EngineError::Other(format!("llama batch: {error}")))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| EngineError::Other(format!("llama decode: {error}")))?;
    }
    Ok(context.get_logits().to_vec())
}

impl LlamaSession {
    pub(crate) fn new(
        model: Arc<LlamaModel>,
        backend: &LlamaBackend,
        context_len: usize,
        offload_kqv: bool,
        residency: Residency,
        inputs: Vec<TensorSpec>,
        outputs: Vec<TensorSpec>,
    ) -> Result<Self> {
        let context_len = context_len.max(1);
        let n_ctx = u32::try_from(context_len).unwrap_or(u32::MAX);
        let batch = u32::try_from(BATCH_TOKENS.min(context_len)).unwrap_or(u32::MAX);
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(batch)
            .with_offload_kqv(offload_kqv);
        let context = ModelContext::try_new(model, |model| {
            model
                .new_context(backend, params)
                .map_err(|error| EngineError::LoadFailed(format!("llama context: {error}")))
        })?;
        Ok(Self {
            state: Mutex::new(LlamaState {
                context,
                cached_ids: Vec::new(),
            }),
            cancelled: AtomicBool::new(false),
            residency,
            inputs,
            outputs,
        })
    }
}
