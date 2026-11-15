//! Loaded ONNX session and execution lifecycle.

use std::sync::Mutex;

use uniinfer_core::{EngineError, EngineSession, Result, Tensor, TensorSpec};

use crate::genai::{GenaiSpec, GenaiState};
use crate::graph::{Graph, GraphCanceller};

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

/// GenAI session: KV cache values are regular tensors over `Graph`.
pub(crate) struct GenaiOnnxSession {
    state: Mutex<GenaiRuntime>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    cancel_handle: GraphCanceller,
}

struct GenaiRuntime {
    graph: Graph,
    genai: GenaiState,
}

impl GenaiOnnxSession {
    pub(crate) fn new(graph: Graph, spec: GenaiSpec) -> Result<Self> {
        let inputs = spec.input_specs(&graph)?;
        let outputs = spec.output_specs(&graph)?;
        let cancel_handle = graph.canceller();
        Ok(Self {
            state: Mutex::new(GenaiRuntime {
                graph,
                genai: GenaiState::new(spec),
            }),
            inputs,
            outputs,
            cancel_handle,
        })
    }
}

impl EngineSession for GenaiOnnxSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("genai session lock poisoned".into()))?;
        let GenaiRuntime { graph, genai } = &mut *state;
        genai.forward(graph, &inputs)
    }

    fn cancel(&self) {
        self.cancel_handle.cancel();
    }
}
