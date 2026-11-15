//! Named ONNX graph wrapper used by multi-graph model adapters.
//!
//! Adapters describe tensors and data flow; ORT `Session`, `DynValue` and
//! output-map handling stay in this module.

use std::path::Path;

use ort::environment::Environment;
use ort::session::Session;
use uniinfer_core::{DeviceClass, EngineError, Result, Tensor, TensorSpec};

use crate::build_session;
use crate::io::{convert_specs, from_ort, to_ort};

pub(crate) struct Graph {
    session: Session,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl Graph {
    pub(crate) fn load(
        environment: &Environment,
        class: DeviceClass,
        path: &Path,
        label: &str,
    ) -> Result<Self> {
        let session = build_session(environment, class, path)?;
        let inputs = convert_specs(session.inputs(), &format!("{label} input"))?;
        let outputs = convert_specs(session.outputs(), &format!("{label} output"))?;
        Ok(Self {
            session,
            inputs,
            outputs,
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

        let values = inputs
            .into_iter()
            .map(|tensor| {
                let name = tensor.name.clone();
                to_ort(tensor).map(|value| (name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self
            .session
            .run(values)
            .map_err(|error| EngineError::Other(format!("onnx graph run: {error}")))?;
        let mut result = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let value = outputs
                .get(spec.name.as_str())
                .ok_or_else(|| EngineError::Other(format!("missing graph output {}", spec.name)))?;
            result.push(from_ort(spec, value)?);
        }
        Ok(result)
    }
}
