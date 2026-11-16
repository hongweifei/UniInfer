//! PCM framing, WAV decoding and energy-based voice activity detection.

use std::io::Cursor;

use uniinfer_core::{EngineError, Result};

pub const SAMPLE_RATE: usize = 16_000;
pub const FRAME_MS: usize = 20;
pub const FRAME_BYTES: usize = SAMPLE_RATE / 1000 * FRAME_MS * 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStarted,
    SpeechStopped,
}

/// Minimal energy VAD with hangover for orchestrated realtime sessions.
#[derive(Debug)]
pub struct EnergyVad {
    threshold: f64,
    speaking: bool,
    silence_frames: u8,
}

impl EnergyVad {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            speaking: false,
            silence_frames: 0,
        }
    }

    pub fn push(&mut self, frame: &[u8]) -> Option<VadEvent> {
        let rms = rms(frame);
        if rms >= self.threshold {
            self.silence_frames = 0;
            if !self.speaking {
                self.speaking = true;
                return Some(VadEvent::SpeechStarted);
            }
            return None;
        }
        if self.speaking {
            self.silence_frames = self.silence_frames.saturating_add(1);
            if self.silence_frames >= 10 {
                self.speaking = false;
                self.silence_frames = 0;
                return Some(VadEvent::SpeechStopped);
            }
        }
        None
    }
}

/// Decode a WAV into normalized mono F32 samples at the expected rate.
///
/// Stereo files are downmixed by averaging channels, and other sample rates
/// are linearly resampled. WAV is still the only container; MP3/Opus decoding
/// belongs to the HTTP ingest layer.
pub fn decode_wav_mono(bytes: &[u8], expected_rate: u32) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|error| EngineError::InvalidAudio(format!("wav: {error}")))?;
    let spec = reader.spec();
    if !(1..=2).contains(&spec.channels) {
        return Err(EngineError::InvalidAudio(format!(
            "expected mono or stereo, got {} channels",
            spec.channels
        )));
    }
    if expected_rate == 0 {
        return Err(EngineError::InvalidAudio(
            "expected sample rate must be positive".into(),
        ));
    }

    let raw = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| value as f32 / scale)
                        .map_err(|error| EngineError::InvalidAudio(format!("wav sample: {error}")))
                })
                .collect::<Result<Vec<_>>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| {
                sample.map_err(|error| EngineError::InvalidAudio(format!("wav sample: {error}")))
            })
            .collect::<Result<Vec<_>>>()?,
    };

    let mono = if spec.channels == 2 {
        raw.chunks_exact(2)
            .map(|frame| (frame[0] + frame[1]) * 0.5)
            .collect::<Vec<_>>()
    } else {
        raw
    };
    Ok(resample(&mono, spec.sample_rate, expected_rate))
}

fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = (samples.len() as u64 * u64::from(to) / u64::from(from)) as usize;
    let mut output = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let position = index as f64 * f64::from(from) / f64::from(to);
        let left = position.floor() as usize;
        let fraction = (position - left as f64) as f32;
        let first = samples[left.min(samples.len() - 1)];
        let second = samples[(left + 1).min(samples.len() - 1)];
        output.push(first + (second - first) * fraction);
    }
    output
}

fn rms(frame: &[u8]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum = frame
        .chunks_exact(2)
        .map(|sample| {
            let value = f64::from(i16::from_le_bytes([sample[0], sample[1]]));
            value * value
        })
        .sum::<f64>();
    (sum / (frame.len() as f64 / 2.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(value: i16) -> Vec<u8> {
        value.to_le_bytes().repeat(FRAME_BYTES / 2)
    }

    fn wav(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
        let mut buffer = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(&mut buffer, spec).expect("writer");
            for &sample in samples {
                writer.write_sample(sample).expect("sample");
            }
            writer.finalize().expect("finalize");
        }
        buffer.into_inner()
    }

    #[test]
    fn decodes_mono_16k_wav() {
        let bytes = wav(&[0, i16::MAX, i16::MIN], 16_000, 1);
        let samples = decode_wav_mono(&bytes, 16_000).expect("decode");
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 1.0).abs() < 1e-4);
        assert!((samples[2] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn downmixes_stereo_and_resamples() {
        let stereo = wav(&[10_000, 20_000, -10_000, -20_000], 32_000, 2);
        let samples = decode_wav_mono(&stereo, 16_000).expect("decode");
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 15_000.0 / 32_768.0).abs() < 1e-4);
    }

    #[test]
    fn rejects_more_than_two_channels() {
        let surround = wav(&[0, 0, 0, 0, 0, 0], 16_000, 3);
        assert!(matches!(
            decode_wav_mono(&surround, 16_000),
            Err(EngineError::InvalidAudio(_))
        ));
    }

    #[test]
    fn detects_speech_with_hangover() {
        let mut vad = EnergyVad::new(500.0);
        assert_eq!(vad.push(&frame(0)), None);
        assert_eq!(vad.push(&frame(2_000)), Some(VadEvent::SpeechStarted));
        for _ in 0..9 {
            assert_eq!(vad.push(&frame(0)), None);
        }
        assert_eq!(vad.push(&frame(0)), Some(VadEvent::SpeechStopped));
    }
}
