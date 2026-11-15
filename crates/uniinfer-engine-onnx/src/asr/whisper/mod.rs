//! ONNX encoder-decoder speech transcription.
//!
//! The model directory is described by the manifest `[audio]` section. The
//! ONNX engine remains the owning engine; this adapter implements the same
//! public `audio` -> `text` session contract as any other ASR engine.

mod cache;
mod config;

use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::{DynValue, Tensor as OrtTensor};
use tokenizers::Tokenizer;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result, Tensor, TensorData,
    TensorDtype, TensorSpec,
};

use crate::build_session;

use self::cache::{run_decoder, run_with_past};
use self::config::{WhisperConfig, options};
use super::frontend::mel;

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<WhisperOnnxSession> {
    let audio = spec
        .manifest
        .audio
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("missing [audio] config".into()))?;
    let options = options(audio)?;
    let encoder = super::options::path(&spec.base_dir, &options.encoder_file, "encoder_file")?;
    let decoder = super::options::path(&spec.base_dir, &options.decoder_file, "decoder_file")?;
    let tokenizer_file =
        super::options::path(&spec.base_dir, &options.tokenizer_file, "tokenizer_file")?;
    let config_file = super::options::path(&spec.base_dir, &options.config_file, "config_file")?;
    spec.manifest.verify_files(&spec.base_dir)?;

    let encoder = build_session(environment, class, &encoder)?;
    let decoder = build_session(environment, class, &decoder)?;
    let decoder_with_past = options
        .decoder_with_past_file
        .as_ref()
        .map(|file| {
            super::options::path(&spec.base_dir, file, "decoder_with_past_file")
                .and_then(|path| build_session(environment, class, &path))
        })
        .transpose()?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|error| EngineError::LoadFailed(format!("tokenizer: {error}")))?;
    let config = WhisperConfig::load(&config_file)?;
    let max_tokens = options
        .max_tokens
        .or(config.max_length)
        .unwrap_or(448)
        .min(config.max_length.unwrap_or(448));

    let input_name = audio.input.clone();
    let output_name = audio.output.clone();
    Ok(WhisperOnnxSession {
        state: std::sync::Mutex::new(WhisperState {
            encoder,
            decoder,
            decoder_with_past,
            tokenizer,
            config,
            sample_rate: audio.sample_rate,
            max_tokens,
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

pub(crate) struct WhisperOnnxSession {
    state: std::sync::Mutex<WhisperState>,
    input_name: String,
    output_name: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    cancelled: AtomicBool,
}

struct WhisperState {
    encoder: Session,
    decoder: Session,
    decoder_with_past: Option<Session>,
    tokenizer: Tokenizer,
    config: WhisperConfig,
    sample_rate: u32,
    max_tokens: usize,
}

impl EngineSession for WhisperOnnxSession {
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
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("whisper onnx state lock poisoned".into()))?;
        let features = mel::log_mel(samples, state.sample_rate)?;
        let text = state.transcribe(&features, &self.cancelled)?;
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

impl WhisperState {
    fn transcribe(&mut self, features: &[f32], cancelled: &AtomicBool) -> Result<String> {
        let encoder_input = self
            .encoder
            .inputs()
            .first()
            .map(|outlet| outlet.name().to_owned())
            .ok_or_else(|| EngineError::LoadFailed("encoder has no inputs".into()))?;
        let encoder_output = self
            .encoder
            .outputs()
            .first()
            .map(|outlet| outlet.name().to_owned())
            .ok_or_else(|| EngineError::LoadFailed("encoder has no outputs".into()))?;
        let features =
            OrtTensor::from_array((vec![1usize, mel::N_MELS, mel::N_FRAMES], features.to_vec()))
                .map_err(|error| EngineError::Other(format!("encoder features: {error}")))?
                .into_dyn();
        let outputs = self
            .encoder
            .run(vec![(encoder_input, features)])
            .map_err(|error| EngineError::Other(format!("onnx encoder: {error}")))?;
        let (hidden_shape, hidden_values) = outputs[encoder_output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|error| EngineError::Other(format!("encoder output: {error}")))?;
        let hidden_shape = hidden_shape
            .iter()
            .map(|dim| *dim as usize)
            .collect::<Vec<_>>();
        let hidden_values = hidden_values.to_vec();

        let mut ids = self.config.initial_ids();
        let mut generated = Vec::new();
        let mut cache: Option<Vec<(String, DynValue)>> = None;
        while generated.len() < self.max_tokens {
            if cancelled.load(Ordering::SeqCst) {
                return Err(EngineError::Cancelled);
            }
            let token = match self.decoder_with_past.as_mut() {
                Some(with_past) => match cache.take() {
                    Some(old_cache) => {
                        let position = ids.len().saturating_sub(1);
                        let (token, new_cache) = run_with_past(
                            with_past,
                            *ids.last().unwrap_or(&self.config.decoder_start_token_id),
                            old_cache,
                            position,
                        )?;
                        cache = Some(new_cache);
                        token
                    }
                    None => {
                        let (token, new_cache) =
                            run_decoder(&mut self.decoder, &ids, &hidden_shape, &hidden_values)?;
                        cache = Some(new_cache);
                        token
                    }
                },
                None => run_decoder(&mut self.decoder, &ids, &hidden_shape, &hidden_values)?.0,
            };
            if token == self.config.eos_token_id {
                break;
            }
            ids.push(token);
            generated.push(token);
        }
        self.tokenizer
            .decode(&generated, true)
            .map_err(|error| EngineError::Other(format!("detokenize: {error}")))
    }
}
