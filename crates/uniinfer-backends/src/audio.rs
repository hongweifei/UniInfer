//! PCM framing and energy-based voice activity detection.

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
