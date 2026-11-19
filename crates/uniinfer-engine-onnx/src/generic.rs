//! Generic ONNX graph execution, independent of model architecture.
//!
//! A manifest selecting `onnx.graph` gets a plain session over the first
//! declared file. If an `onnxruntime-genai` config is present the same graph is
//! wrapped with KV-cache generation.

use ort::environment::Environment;
use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result};

use crate::build_session;
use crate::genai::{GenaiSpec, load_genai_config};
use crate::graph::Graph;
use crate::session::{GenaiOnnxSession, PlainOnnxSession};

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let model_file = spec
        .manifest
        .files
        .first()
        .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
    let model_path = spec.base_dir.join(&model_file.path);
    let session = build_session(
        environment,
        class,
        &model_path,
        crate::backend_variant(spec),
    )?;
    if spec.cancel.is_cancelled() {
        return Err(EngineError::Cancelled);
    }
    let graph = Graph::from_session(session, "onnx")?;
    if let Some(config) = load_genai_config(&spec.base_dir)? {
        let genai = GenaiSpec::from_config(&config, &graph)?;
        genai.input_specs(&graph)?;
        genai.output_specs(&graph)?;
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        return Ok(Box::new(GenaiOnnxSession::new(graph, genai)?));
    }
    tracing::debug!(
        model = %spec.manifest.id,
        path = %model_path.display(),
        inputs = graph.inputs().len(),
        outputs = graph.outputs().len(),
        "generic onnx session loaded"
    );
    if spec.cancel.is_cancelled() {
        return Err(EngineError::Cancelled);
    }
    Ok(Box::new(PlainOnnxSession::new(graph)))
}
