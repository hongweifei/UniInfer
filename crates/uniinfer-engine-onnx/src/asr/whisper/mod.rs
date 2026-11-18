//! ONNX encoder-decoder speech transcription.
//!
//! The model directory is described by the manifest `[audio]` section. The
//! ONNX engine remains the owning engine; this adapter implements the same
//! public `audio` -> `text` session contract as any other ASR engine.

mod cache;
mod config;

use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use tokenizers::Tokenizer;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result, Tensor, TensorData,
    TensorDtype, TensorSpec,
};

use crate::graph::{Graph, GraphCanceller};

use self::cache::{run_decoder, run_with_past};
use self::config::{WhisperConfig, options};
use super::frontend::mel;

/// One second of shared audio between consecutive 30 s Whisper windows.
const CHUNK_OVERLAP: usize = 16_000;

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

    let encoder = Graph::load(
        environment,
        class,
        &encoder,
        "whisper encoder",
        crate::backend_variant(spec),
    )?;
    let decoder = Graph::load(
        environment,
        class,
        &decoder,
        "whisper decoder",
        crate::backend_variant(spec),
    )?;
    let decoder_with_past = options
        .decoder_with_past_file
        .as_ref()
        .map(|file| {
            super::options::path(&spec.base_dir, file, "decoder_with_past_file").and_then(|path| {
                Graph::load(
                    environment,
                    class,
                    &path,
                    "whisper decoder_with_past",
                    crate::backend_variant(spec),
                )
            })
        })
        .transpose()?;
    let cancels = [
        Some(encoder.canceller()),
        Some(decoder.canceller()),
        decoder_with_past.as_ref().map(Graph::canceller),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
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
        cancels,
    })
}

pub(crate) struct WhisperOnnxSession {
    state: std::sync::Mutex<WhisperState>,
    input_name: String,
    output_name: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    cancelled: AtomicBool,
    cancels: Vec<GraphCanceller>,
}

struct WhisperState {
    encoder: Graph,
    decoder: Graph,
    decoder_with_past: Option<Graph>,
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
        let text = state.transcribe_audio(samples, &self.cancelled)?;
        let bytes = text.trim().as_bytes().to_vec();
        Ok(vec![Tensor::new(
            self.output_name.clone(),
            vec![bytes.len()],
            TensorData::U8(bytes),
        )?])
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        for cancel in &self.cancels {
            cancel.cancel();
        }
    }
}

impl WhisperState {
    fn transcribe_audio(&mut self, samples: &[f32], cancelled: &AtomicBool) -> Result<String> {
        if samples.is_empty() {
            return Err(EngineError::InvalidAudio("audio is empty".into()));
        }
        let mut transcript = String::new();
        let mut start = 0usize;
        loop {
            if cancelled.load(Ordering::SeqCst) {
                return Err(EngineError::Cancelled);
            }
            let end = (start + mel::N_SAMPLES).min(samples.len());
            let features = mel::log_mel(&samples[start..end], self.sample_rate)?;
            let text = self.transcribe_features(&features, cancelled)?;
            merge_overlap(&mut transcript, text.trim());
            if end == samples.len() {
                break;
            }
            start = end - CHUNK_OVERLAP.min(end);
        }
        Ok(transcript)
    }

    fn transcribe_features(&mut self, features: &[f32], cancelled: &AtomicBool) -> Result<String> {
        let encoder_input = self
            .encoder
            .inputs()
            .first()
            .map(|spec| spec.name.clone())
            .ok_or_else(|| EngineError::LoadFailed("encoder has no inputs".into()))?;
        let features = Tensor::new(
            encoder_input,
            vec![1, mel::N_MELS, mel::N_FRAMES],
            TensorData::F32(features.to_vec()),
        )?;
        let hidden = self
            .encoder
            .run(vec![features])?
            .into_iter()
            .next()
            .ok_or_else(|| EngineError::LoadFailed("encoder has no outputs".into()))?;

        let mut ids = self.config.initial_ids();
        let mut generated = Vec::new();
        let mut cache: Option<Vec<Tensor>> = None;
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
                        let (token, new_cache) = run_decoder(&mut self.decoder, &ids, &hidden)?;
                        cache = Some(new_cache);
                        token
                    }
                },
                None => run_decoder(&mut self.decoder, &ids, &hidden)?.0,
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

/// Append `next` to `transcript`, removing the longest word-level suffix/prefix
/// shared by the one-second audio overlap.
fn merge_overlap(transcript: &mut String, next: &str) {
    if next.is_empty() {
        return;
    }
    let existing = transcript.split_whitespace().collect::<Vec<_>>();
    let incoming = next.split_whitespace().collect::<Vec<_>>();
    if existing.is_empty() {
        transcript.push_str(next);
        return;
    }
    let max = existing.len().min(incoming.len());
    let overlap = (1..=max)
        .rev()
        .find(|&count| existing[existing.len() - count..] == incoming[..count])
        .unwrap_or(0);
    for word in &incoming[overlap..] {
        if !transcript.is_empty() {
            transcript.push(' ');
        }
        transcript.push_str(word);
    }
}

#[cfg(test)]
mod tests {
    use super::merge_overlap;

    #[test]
    fn merges_word_overlap_without_duplicate() {
        let mut transcript = String::from("the quick brown fox");
        merge_overlap(&mut transcript, "brown fox jumps over");
        assert_eq!(transcript, "the quick brown fox jumps over");
    }

    #[test]
    fn keeps_disjoint_chunks() {
        let mut transcript = String::from("hello world");
        merge_overlap(&mut transcript, "second window");
        assert_eq!(transcript, "hello world second window");
    }
}
