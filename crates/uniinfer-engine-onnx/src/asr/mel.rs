//! Whisper-compatible log-Mel frontend.
//!
//! Matches the reference feature extractor: 16 kHz mono, 30 s window,
//! 400-point FFT, 160-sample hop, 80 Slaney mel filters, log10 with a
//! dynamic range floor, then the `(x + 4) / 4` normalization.

use std::f64::consts::PI;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use uniinfer_core::{EngineError, Result};

pub(crate) const SAMPLE_RATE: u32 = 16_000;
pub(crate) const N_FFT: usize = 400;
pub(crate) const HOP_LENGTH: usize = 160;
pub(crate) const N_MELS: usize = 80;
pub(crate) const N_FRAMES: usize = 3_000;
pub(crate) const N_SAMPLES: usize = 480_000;
const N_FREQS: usize = N_FFT / 2 + 1;

/// Produce `[80, 3000]` features in mel-major order.
pub(crate) fn log_mel(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    if sample_rate != SAMPLE_RATE {
        return Err(EngineError::InvalidAudio(format!(
            "whisper requires {SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
        )));
    }
    if samples.is_empty() {
        return Err(EngineError::InvalidAudio("audio is empty".into()));
    }

    let mut audio = vec![0.0f32; N_SAMPLES];
    let copy = samples.len().min(N_SAMPLES);
    audio[..copy].copy_from_slice(&samples[..copy]);

    // torch.stft(center=True) pads by n_fft / 2 with reflection.
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; N_SAMPLES + 2 * pad];
    padded[pad..pad + N_SAMPLES].copy_from_slice(&audio);
    for index in 1..=pad {
        padded[pad - index] = audio[index];
        padded[pad + N_SAMPLES + index - 1] = audio[N_SAMPLES - index - 1];
    }

    let window = (0..N_FFT)
        .map(|index| 0.5 * (1.0 - (2.0 * PI * index as f64 / N_FFT as f64).cos()) as f32)
        .collect::<Vec<_>>();
    let now = std::time::Instant::now();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];
    let mut mel = vec![0.0f32; N_MELS * N_FRAMES];
    let filters = mel_filters();
    let mut max = f32::NEG_INFINITY;

    for frame in 0..N_FRAMES {
        let start = frame * HOP_LENGTH;
        for index in 0..N_FFT {
            buffer[index] = Complex::new(padded[start + index] * window[index], 0.0);
        }
        fft.process(&mut buffer);
        for mel_index in 0..N_MELS {
            let mut sum = 0.0f32;
            let row = &filters[mel_index * N_FREQS..(mel_index + 1) * N_FREQS];
            for (bin, weight) in row.iter().enumerate() {
                sum += weight * buffer[bin].norm_sqr();
            }
            let value = sum.max(1e-10).log10();
            mel[mel_index * N_FRAMES + frame] = value;
            max = max.max(value);
        }
    }

    let floor = max - 8.0;
    for value in &mut mel {
        *value = ((*value).max(floor) + 4.0) / 4.0;
    }
    tracing::debug!(
        elapsed_ms = now.elapsed().as_secs_f64() * 1000.0,
        "whisper log-mel computed"
    );
    Ok(mel)
}

fn mel_filters() -> Vec<f32> {
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let mel_points = (0..N_MELS + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f64 / (N_MELS + 1) as f64)
        .map(mel_to_hz)
        .collect::<Vec<_>>();
    let fft_freqs = (0..N_FREQS)
        .map(|index| index as f64 * (SAMPLE_RATE as f64 / 2.0) / (N_FREQS - 1) as f64)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; N_MELS * N_FREQS];
    for mel_index in 0..N_MELS {
        let lower = mel_points[mel_index];
        let center = mel_points[mel_index + 1];
        let upper = mel_points[mel_index + 2];
        for (bin, frequency) in fft_freqs.iter().enumerate() {
            let rising = (frequency - lower) / (center - lower);
            let falling = (upper - frequency) / (upper - center);
            let weight = rising.min(falling).max(0.0);
            // Slaney normalization.
            filters[mel_index * N_FREQS + bin] = (weight * 2.0 / (upper - lower)) as f32;
        }
    }
    filters
}

fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let log_step = (6.4f64).ln() / 27.0;
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / log_step
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let log_step = (6.4f64).ln() / 27.0;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * (log_step * (mel - MIN_LOG_MEL)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_expected_shape_and_finite_values() {
        let samples = vec![0.0f32; SAMPLE_RATE as usize / 10];
        let features = log_mel(&samples, SAMPLE_RATE).expect("mel");
        assert_eq!(features.len(), N_MELS * N_FRAMES);
        assert!(features.iter().all(|value| value.is_finite()));
        assert!(features.iter().any(|value| *value < 0.0));
    }

    #[test]
    fn requires_16khz() {
        let error = log_mel(&[0.0], 8_000).expect_err("rate");
        assert!(matches!(error, EngineError::InvalidAudio(_)));
    }
}
