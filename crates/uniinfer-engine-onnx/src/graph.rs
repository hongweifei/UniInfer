//! Named ONNX graph wrapper used by multi-graph model adapters.
//!
//! Adapters describe tensors and data flow; ORT `Session`, `DynValue` and
//! output-map handling stay in this module.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ort::environment::Environment;
use ort::session::{NoSelectedOutputs, RunOptions, Session};
use uniinfer_core::{DeviceClass, EngineError, Result, Tensor, TensorSpec};

use crate::build_session;
use crate::io::{convert_specs, from_ort, to_ort};

#[derive(Clone)]
pub(crate) struct GraphCanceller {
    cancelled: Arc<AtomicBool>,
    active: Arc<Mutex<Option<Arc<RunOptions<NoSelectedOutputs>>>>>,
}

impl GraphCanceller {
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(active) = self.active.lock()
            && let Some(options) = active.as_ref()
        {
            let _ = options.terminate();
        }
    }
}

pub(crate) struct Graph {
    session: Session,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    control: GraphCanceller,
}

impl Graph {
    pub(crate) fn load(
        environment: &Environment,
        class: DeviceClass,
        path: &Path,
        label: &str,
        variant: Option<&str>,
    ) -> Result<Self> {
        let session = build_session(environment, class, path, variant)?;
        Self::from_session(session, label)
    }

    pub(crate) fn from_bytes(
        environment: &Environment,
        class: DeviceClass,
        bytes: &[u8],
        label: &str,
        variant: Option<&str>,
    ) -> Result<Self> {
        let session = crate::build_session_from_memory(environment, class, bytes, variant)?;
        Self::from_session(session, label)
    }

    pub(crate) fn from_session(session: Session, label: &str) -> Result<Self> {
        let inputs = convert_specs(session.inputs(), &format!("{label} input"))?;
        let outputs = convert_specs(session.outputs(), &format!("{label} output"))?;
        Ok(Self {
            session,
            inputs,
            outputs,
            control: GraphCanceller {
                cancelled: Arc::new(AtomicBool::new(false)),
                active: Arc::new(Mutex::new(None)),
            },
        })
    }

    pub(crate) fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    pub(crate) fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    pub(crate) fn run(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        for input in &inputs {
            let spec = self
                .inputs
                .iter()
                .find(|spec| spec.name == input.name)
                .ok_or_else(|| EngineError::Other(format!("unknown graph input {}", input.name)))?;
            if spec.dtype != input.data.dtype() {
                return Err(EngineError::Other(format!(
                    "graph input {} expects {:?}, got {:?}",
                    input.name,
                    spec.dtype,
                    input.data.dtype()
                )));
            }
        }
        for spec in &self.inputs {
            if !inputs.iter().any(|input| input.name == spec.name) {
                return Err(EngineError::Other(format!(
                    "missing graph input {}",
                    spec.name
                )));
            }
        }

        if self.control.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let values = inputs
            .into_iter()
            .map(|tensor| {
                let name = tensor.name.clone();
                to_ort(tensor).map(|value| (name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let options = Arc::new(
            RunOptions::new()
                .map_err(|error| EngineError::Other(format!("onnx run options: {error}")))?,
        );
        *self
            .control
            .active
            .lock()
            .map_err(|_| EngineError::Other("run options lock poisoned".into()))? =
            Some(Arc::clone(&options));
        let outputs = self.session.run_with_options(values, &options);
        if let Ok(mut active) = self.control.active.lock() {
            *active = None;
        }
        let outputs = outputs.map_err(|error| {
            if self.control.cancelled.load(Ordering::SeqCst) {
                EngineError::Cancelled
            } else {
                EngineError::Other(format!("onnx graph run: {error}"))
            }
        })?;
        let mut result = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let value = outputs
                .get(spec.name.as_str())
                .ok_or_else(|| EngineError::Other(format!("missing graph output {}", spec.name)))?;
            result.push(from_ort(spec, value)?);
        }
        Ok(result)
    }

    pub(crate) fn canceller(&self) -> GraphCanceller {
        self.control.clone()
    }
}
