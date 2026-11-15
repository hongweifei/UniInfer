//! ONNX encoder-decoder speech transcription.
//!
//! The model directory is described by the manifest `[audio]` section. The
//! ONNX engine remains the owning engine; this adapter implements the same
//! public `audio` -> `text` session contract as any other ASR engine.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor as OrtTensor};
use serde::Deserialize;
use tokenizers::Tokenizer;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result, Tensor, TensorData,
    TensorDtype, TensorSpec,
};

use crate::build_session;

use super::mel;

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
    let encoder = audio_file(
        &spec.base_dir,
        audio.encoder_file.as_deref(),
        "encoder_file",
    )?;
    let decoder = audio_file(
        &spec.base_dir,
        audio.decoder_file.as_deref(),
        "decoder_file",
    )?;
    let tokenizer_file = audio_file(
        &spec.base_dir,
        audio.tokenizer_file.as_deref(),
        "tokenizer_file",
    )?;
    let config_file = audio_file(&spec.base_dir, audio.config_file.as_deref(), "config_file")?;
    spec.manifest.verify_files(&spec.base_dir)?;

    let encoder = build_session(environment, class, &encoder)?;
    let decoder = build_session(environment, class, &decoder)?;
    let decoder_with_past = audio
        .decoder_with_past_file
        .as_deref()
        .map(|file| build_session(environment, class, &spec.base_dir.join(file)))
        .transpose()?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|error| EngineError::LoadFailed(format!("tokenizer: {error}")))?;
    let config = WhisperConfig::load(&config_file)?;
    let max_tokens = audio
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

fn audio_file(base: &Path, file: Option<&Path>, field: &str) -> Result<std::path::PathBuf> {
    let file = file.ok_or_else(|| {
        EngineError::ManifestInvalid(format!("audio.{field} is required for ONNX ASR"))
    })?;
    Ok(base.join(file))
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

fn run_decoder(
    session: &mut Session,
    ids: &[u32],
    hidden_shape: &[usize],
    hidden_values: &[f32],
) -> Result<(u32, Vec<(String, DynValue)>)> {
    let input_ids_name = session
        .inputs()
        .first()
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no inputs".into()))?;
    let hidden_name = session
        .inputs()
        .get(1)
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no encoder input".into()))?;
    let logits_name = session
        .outputs()
        .first()
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder has no outputs".into()))?;
    let input_ids = OrtTensor::from_array((
        vec![1usize, ids.len()],
        ids.iter().map(|id| i64::from(*id)).collect::<Vec<_>>(),
    ))
    .map_err(|error| EngineError::Other(format!("decoder ids: {error}")))?
    .into_dyn();
    let hidden = OrtTensor::from_array((hidden_shape.to_vec(), hidden_values.to_vec()))
        .map_err(|error| EngineError::Other(format!("encoder hidden: {error}")))?
        .into_dyn();
    let present_names = present_names(session);
    let mut outputs = session
        .run(vec![(input_ids_name, input_ids), (hidden_name, hidden)])
        .map_err(|error| EngineError::Other(format!("onnx decoder: {error}")))?;
    let token = token_from_logits(&outputs, &logits_name)?;
    let cache = take_present(&mut outputs, &present_names)?;
    Ok((token, cache))
}

fn run_with_past(
    session: &mut Session,
    token: u32,
    cache: Vec<(String, DynValue)>,
    position: usize,
) -> Result<(u32, Vec<(String, DynValue)>)> {
    let input_ids_name = session
        .inputs()
        .iter()
        .find(|outlet| outlet.name() == "input_ids")
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder_with_past has no input_ids".into()))?;
    let logits_name = session
        .outputs()
        .first()
        .map(|outlet| outlet.name().to_owned())
        .ok_or_else(|| EngineError::LoadFailed("decoder_with_past has no outputs".into()))?;
    let input_names = session
        .inputs()
        .iter()
        .map(|outlet| outlet.name().to_owned())
        .collect::<Vec<_>>();

    let ids_value = OrtTensor::from_array((vec![1usize, 1usize], vec![i64::from(token)]))
        .map_err(|error| EngineError::Other(format!("past ids: {error}")))?
        .into_dyn();
    let position_value = OrtTensor::from_array((vec![1usize], vec![position as i64]))
        .map_err(|error| EngineError::Other(format!("cache position: {error}")))?
        .into_dyn();
    let mut values: Vec<(String, SessionInputValue<'_>)> = Vec::with_capacity(input_names.len());
    for name in input_names {
        match name.as_str() {
            "input_ids" => {
                values.push((input_ids_name.clone(), SessionInputValue::from(&ids_value)));
            }
            "cache_position" => {
                values.push((name, SessionInputValue::from(&position_value)));
            }
            _ => {
                let present = name.replacen("past_key_values.", "present.", 1);
                let value = cache
                    .iter()
                    .find(|(cached, _)| cached == &present)
                    .map(|(_, value)| value)
                    .ok_or_else(|| EngineError::Other(format!("missing past cache {present}")))?;
                values.push((name, SessionInputValue::from(value)));
            }
        }
    }
    let present_names = present_names(session);
    let mut outputs = session
        .run(values)
        .map_err(|error| EngineError::Other(format!("onnx decoder_with_past: {error}")))?;
    let token = token_from_logits(&outputs, &logits_name)?;
    let retained = cache
        .into_iter()
        .filter(|(name, _)| name.contains(".encoder."))
        .collect::<Vec<_>>();
    let mut cache = take_present(&mut outputs, &present_names)?;
    cache.extend(retained);
    Ok((token, cache))
}

fn token_from_logits(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<u32> {
    let (shape, logits) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|error| EngineError::Other(format!("decoder logits: {error}")))?;
    let shape = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    let vocab = *shape
        .last()
        .ok_or_else(|| EngineError::Other("empty logits shape".into()))?;
    let offset = logits.len().saturating_sub(vocab);
    Ok(argmax(&logits[offset..]))
}

fn present_names(session: &Session) -> Vec<String> {
    session
        .outputs()
        .iter()
        .map(|outlet| outlet.name().to_owned())
        .filter(|name| name.starts_with("present."))
        .collect()
}

fn take_present(
    outputs: &mut ort::session::SessionOutputs<'_>,
    names: &[String],
) -> Result<Vec<(String, DynValue)>> {
    let mut values = Vec::with_capacity(names.len());
    for name in names {
        let value = outputs
            .remove(name.as_str())
            .ok_or_else(|| EngineError::Other(format!("missing present cache {name}")))?;
        values.push((name.clone(), value));
    }
    Ok(values)
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if value.is_finite() && *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    u32::try_from(best).unwrap_or(u32::MAX)
}

#[derive(Debug, Clone, Deserialize)]
struct WhisperConfig {
    decoder_start_token_id: u32,
    eos_token_id: u32,
    #[serde(default)]
    forced_decoder_ids: Vec<[u32; 2]>,
    #[serde(default)]
    max_length: Option<usize>,
}

impl WhisperConfig {
    fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            EngineError::ManifestInvalid(format!("cannot read {}: {error}", path.display()))
        })?;
        let config: Self = serde_json::from_str(&raw).map_err(|error| {
            EngineError::ManifestInvalid(format!("{}: {error}", path.display()))
        })?;
        Ok(config)
    }

    fn initial_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.decoder_start_token_id];
        ids.extend(self.forced_decoder_ids.iter().map(|pair| pair[1]));
        ids
    }
}
