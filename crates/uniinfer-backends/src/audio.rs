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

/// Minimal energy VAD with hangover, matching the M3 orchestrated baseline.
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

/// Decode a mono 16 kHz PCM WAV into normalized F32 samples.
pub fn decode_wav_16k_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|error| EngineError::InvalidAudio(format!("wav: {error}")))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(EngineError::InvalidAudio(format!(
            "expected mono, got {} channels",
            spec.channels
        )));
    }
    if spec.sample_rate != SAMPLE_RATE as u32 {
        return Err(EngineError::InvalidAudio(format!(
            "expected {} Hz, got {} Hz",
            SAMPLE_RATE, spec.sample_rate
        )));
    }

    let samples = match spec.sample_format {
        hound::SampleFormat::Int if spec.bits_per_sample == 16 => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| f32::from(value) / 32768.0)
                    .map_err(|error| EngineError::InvalidAudio(format!("wav sample: {error}")))
            })
            .collect::<Result<Vec<_>>>()?,
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
    Ok(samples)
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
        let samples = decode_wav_16k_mono(&bytes).expect("decode");
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 1.0).abs() < 1e-4);
        assert!((samples[2] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn rejects_stereo_and_wrong_rate() {
        let stereo = wav(&[0, 0], 16_000, 2);
        assert!(matches!(
            decode_wav_16k_mono(&stereo),
            Err(EngineError::InvalidAudio(_))
        ));
        let rate = wav(&[0], 44_100, 1);
        assert!(matches!(
            decode_wav_16k_mono(&rate),
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
