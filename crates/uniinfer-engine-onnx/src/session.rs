//! Loaded ONNX session and execution lifecycle.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::session::{NoSelectedOutputs, RunOptions, Session};
use ort::value::DynValue;
use uniinfer_core::{EngineError, EngineSession, Result, Tensor, TensorSpec};

use crate::genai::{GenaiState, run_genai};
use crate::io::{from_ort, to_ort};

pub(crate) struct OnnxState {
    pub(crate) session: Session,
    pub(crate) genai: Option<GenaiState>,
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
        if state.genai.is_some() {
            return run_genai(&mut state, &inputs, &active.options, &self.cancelled);
        }

        let mut values: Vec<(String, DynValue)> = Vec::with_capacity(inputs.len());
        for input in inputs {
            if !self.inputs.iter().any(|spec| spec.name == input.name) {
                return Err(EngineError::Other(format!(
                    "unknown input tensor {}",
                    input.name
                )));
            }
            values.push((input.name.clone(), to_ort(input)?));
        }

        for spec in &self.inputs {
            if !values.iter().any(|(name, _)| name == &spec.name) {
                return Err(EngineError::Other(format!(
                    "missing input tensor {}",
                    spec.name
                )));
            }
        }

        let OnnxState { session, .. } = &mut *state;
        let outputs = session
            .run_with_options(values, &active.options)
            .map_err(|error| {
                if self.cancelled.load(Ordering::SeqCst) {
                    EngineError::Cancelled
                } else {
                    EngineError::Other(format!("onnx run: {error}"))
                }
            })?;

        let mut result = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let value = &outputs[spec.name.as_str()];
            result.push(from_ort(spec, value)?);
        }
        Ok(result)
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
