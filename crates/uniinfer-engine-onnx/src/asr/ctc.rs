//! Generic single-graph CTC ASR adapter.
//!
//! This covers ONNX ASR models that consume 16 kHz raw audio and emit a
//! `[batch, frames, vocab]` logits tensor (wav2vec2/Conformer-style models).
//! Model-specific tokenizer, graph tensor names and blank id stay in the
//! manifest; no model family is compiled into the engine.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use serde::Deserialize;
use tokenizers::Tokenizer;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result, Tensor, TensorData,
    TensorDtype, TensorSpec,
};

use crate::graph::Graph;

use super::decode::ctc_tokens;
use super::options;

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<CtcOnnxSession> {
    let audio = spec
        .manifest
        .audio
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("missing [audio] config".into()))?;
    let options: CtcOptions = options::parse(audio)?;
    if options.frontend != "raw" {
        return Err(EngineError::UnsupportedEngine(format!(
            "CTC frontend {:?}; only raw audio is registered",
            options.frontend
        )));
    }
    let graph = match &options.graph_file {
        Some(file) => options::path(&spec.base_dir, file, "graph_file")?,
        None => spec
            .manifest
            .files
            .first()
            .map(|file| options::path(&spec.base_dir, &file.path, "graph_file"))
            .transpose()?
            .ok_or_else(|| EngineError::ManifestInvalid("audio.graph_file is required".into()))?,
    };
    let tokenizer_file = options::path(&spec.base_dir, &options.tokenizer_file, "tokenizer_file")?;
    spec.manifest.verify_files(&spec.base_dir)?;
    let graph = Graph::load(environment, class, &graph, "ctc")?;
    if !graph
        .inputs()
        .iter()
        .any(|input| input.name == options.graph_input)
    {
        return Err(EngineError::ManifestInvalid(format!(
            "CTC graph has no input {:?}",
            options.graph_input
        )));
    }
    if !graph
        .outputs()
        .iter()
        .any(|output| output.name == options.graph_output)
    {
        return Err(EngineError::ManifestInvalid(format!(
            "CTC graph has no output {:?}",
            options.graph_output
        )));
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|error| EngineError::LoadFailed(format!("tokenizer: {error}")))?;
    let input_name = audio.input.clone();
    let output_name = audio.output.clone();
    Ok(CtcOnnxSession {
        state: Mutex::new(CtcState {
            graph,
            tokenizer,
            graph_input: options.graph_input.clone(),
            graph_output: options.graph_output.clone(),
            blank_token: options.blank_token,
        }),
        input_name: input_name.clone(),
        output_name: output_name.clone(),
        inputs: vec![TensorSpec {
            name: input_name,
            dtype: TensorDtype::F32,
            shape: vec![None],
        }],
        outputs: vec![TensorSpec {
            name: output_name,
            dtype: TensorDtype::U8,
            shape: vec![None],
        }],
        cancelled: AtomicBool::new(false),
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CtcOptions {
    #[serde(default)]
    graph_file: Option<std::path::PathBuf>,
    #[serde(default = "default_graph_input")]
    graph_input: String,
    #[serde(default = "default_graph_output")]
    graph_output: String,
    tokenizer_file: std::path::PathBuf,
    #[serde(default = "default_frontend")]
    frontend: String,
    #[serde(default)]
    blank_token: u32,
}

fn default_graph_input() -> String {
    "input_values".into()
}

fn default_graph_output() -> String {
    "logits".into()
}

fn default_frontend() -> String {
    "raw".into()
}

pub(crate) struct CtcOnnxSession {
    state: Mutex<CtcState>,
    input_name: String,
    output_name: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    cancelled: AtomicBool,
}

struct CtcState {
    graph: Graph,
    tokenizer: Tokenizer,
    graph_input: String,
    graph_output: String,
    blank_token: u32,
}

impl EngineSession for CtcOnnxSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let input = inputs
            .iter()
            .find(|tensor| tensor.name == self.input_name)
            .ok_or_else(|| EngineError::InvalidAudio("missing audio input".into()))?;
        let TensorData::F32(samples) = &input.data else {
            return Err(EngineError::InvalidAudio("audio must be f32".into()));
        };
        if samples.is_empty() {
            return Err(EngineError::InvalidAudio("audio is empty".into()));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("CTC state lock poisoned".into()))?;
        let graph_input = state.graph_input.clone();
        let graph_output = state.graph_output.clone();
        let (shape, logits) = {
            let value = Tensor::new(
                graph_input.clone(),
                vec![1, samples.len()],
                TensorData::F32(samples.clone()),
            )?;
            let outputs = state.graph.run(vec![value])?;
            let logits = outputs
                .into_iter()
                .find(|tensor| tensor.name == graph_output)
                .ok_or_else(|| EngineError::Other("CTC logits missing".into()))?;
            let TensorData::F32(values) = logits.data else {
                return Err(EngineError::Other("CTC logits must be f32".into()));
            };
            (logits.shape, values)
        };
        let vocab = *shape
            .last()
            .ok_or_else(|| EngineError::Other("empty CTC logits shape".into()))?;
        let frames = logits.len() / vocab;
        let tokens = ctc_tokens(&logits, frames, vocab, state.blank_token);
        let text = state
            .tokenizer
            .decode(&tokens, true)
            .map_err(|error| EngineError::Other(format!("detokenize: {error}")))?;
        let bytes = text.trim().as_bytes().to_vec();
        Ok(vec![Tensor::new(
            self.output_name.clone(),
            vec![bytes.len()],
            TensorData::U8(bytes),
        )?])
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}
