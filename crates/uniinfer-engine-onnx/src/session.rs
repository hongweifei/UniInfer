//! Loaded ONNX session and execution lifecycle.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::session::{NoSelectedOutputs, RunOptions, Session};
use uniinfer_core::{EngineError, EngineSession, Result, Tensor, TensorSpec};

use crate::genai::{GenaiState, run_genai};
use crate::graph::{Graph, GraphCanceller};

pub(crate) struct OnnxState {
    pub(crate) session: Session,
    pub(crate) genai: Option<GenaiState>,
}

/// Ordinary single-graph ONNX session backed by the shared `Graph` layer.
pub(crate) struct PlainOnnxSession {
    graph: Mutex<Graph>,
    cancel_handle: GraphCanceller,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl PlainOnnxSession {
    pub(crate) fn new(graph: Graph) -> Self {
        let inputs = graph.inputs().to_vec();
        let outputs = graph.outputs().to_vec();
        let cancel_handle = graph.canceller();
        Self {
            graph: Mutex::new(graph),
            cancel_handle,
            inputs,
            outputs,
        }
    }
}

impl EngineSession for PlainOnnxSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.graph
            .lock()
            .map_err(|_| EngineError::Other("graph lock poisoned".into()))?
            .run(inputs)
    }

    fn cancel(&self) {
        self.cancel_handle.cancel();
    }
}

pub(crate) struct OnnxSession {
    pub(crate) state: Mutex<OnnxState>,
    pub(crate) inputs: Vec<TensorSpec>,
    pub(crate) outputs: Vec<TensorSpec>,
    pub(crate) cancelled: AtomicBool,
    pub(crate) active: Mutex<Option<std::sync::Arc<RunOptions<NoSelectedOutputs>>>>,
}

struct ActiveRun<'a> {
    slot: &'a Mutex<Option<std::sync::Arc<RunOptions<NoSelectedOutputs>>>>,
    options: std::sync::Arc<RunOptions<NoSelectedOutputs>>,
}

impl Drop for ActiveRun<'_> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = None;
        }
    }
}

impl OnnxSession {
    fn begin_run(&self) -> Result<ActiveRun<'_>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let options = std::sync::Arc::new(
            RunOptions::new()
                .map_err(|error| EngineError::Other(format!("onnx run options: {error}")))?,
        );
        *self
            .active
            .lock()
            .map_err(|_| EngineError::Other("run options lock poisoned".into()))? =
            Some(std::sync::Arc::clone(&options));
        Ok(ActiveRun {
            slot: &self.active,
            options,
        })
    }
}

impl EngineSession for OnnxSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let active = self.begin_run()?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("session lock poisoned".into()))?;
        run_genai(&mut state, &inputs, &active.options, &self.cancelled)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(slot) = self.active.lock()
            && let Some(options) = slot.as_ref()
        {
            let _ = options.terminate();
        }
    }
}
