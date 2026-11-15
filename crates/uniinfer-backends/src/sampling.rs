//! Token sampling primitives kept separate from the generation loop.

use uniinfer_core::{EngineError, Result};

/// Decoding controls for one generation request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// `0` means greedy decoding; positive values divide logits.
    pub temperature: f32,
    /// Nucleus probability mass; `1.0` disables nucleus filtering.
    pub top_p: f32,
    /// Keep the best K candidates; `0` keeps every finite token.
    pub top_k: usize,
    /// Deterministic sampling seed; `None` seeds from the clock.
    pub seed: Option<u64>,
    /// Frequency penalty applied to already-generated tokens, `0..=2`.
    pub frequency_penalty: f32,
    /// Presence penalty applied once per already-generated token, `0..=2`.
    pub presence_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl Sampling {
    #[must_use]
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(EngineError::InvalidSampling(
                "temperature must be between 0 and 2".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(EngineError::InvalidSampling(
                "top_p must be in (0, 1]".into(),
            ));
        }
        if self.top_k > 4096 {
            return Err(EngineError::InvalidSampling(
                "top_k must be at most 4096".into(),
            ));
        }
        if !self.frequency_penalty.is_finite()
            || !(0.0..=2.0).contains(&self.frequency_penalty)
            || !self.presence_penalty.is_finite()
            || !(0.0..=2.0).contains(&self.presence_penalty)
        {
            return Err(EngineError::InvalidSampling(
                "frequency/presence penalties must be between 0 and 2".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn sample(
    row: &[f32],
    sampling: &Sampling,
    rng: &mut Rng,
    generated: &[u32],
) -> Result<u32> {
    let mut counts = std::collections::HashMap::<usize, f32>::new();
    if sampling.frequency_penalty > 0.0 || sampling.presence_penalty > 0.0 {
        for token in generated {
            *counts.entry(*token as usize).or_default() += 1.0;
        }
    }
    let penalty = |index: usize| {
        counts.get(&index).map_or(0.0, |count| {
            sampling.frequency_penalty * *count + sampling.presence_penalty
        })
    };
    if sampling.temperature <= 0.0 || row.is_empty() {
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in row.iter().enumerate() {
            if value.is_finite() && *value - penalty(index) > best_value {
                best_value = *value - penalty(index);
                best = index;
            }
        }
        return u32::try_from(best).map_err(|_| EngineError::Other("token id overflow".into()));
    }

    let max = row
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| value - penalty(index))
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(EngineError::Other("sampling logits are not finite".into()));
    }
    let temperature = sampling.temperature;
    let mut candidates = row
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (((value - penalty(index) - max) / temperature).exp(), index))
        .collect::<Vec<_>>();
    if sampling.top_k > 0 && candidates.len() > sampling.top_k {
        candidates.select_nth_unstable_by(sampling.top_k - 1, |a, b| b.0.total_cmp(&a.0));
        candidates.truncate(sampling.top_k);
    }
    candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let mut total = candidates.iter().map(|(weight, _)| weight).sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(EngineError::Other("sampling weights are not finite".into()));
    }
    if sampling.top_p < 1.0 {
        let cutoff = total * sampling.top_p;
        let mut cumulative = 0.0;
        let keep = candidates
            .iter()
            .position(|(weight, _)| {
                cumulative += weight;
                cumulative >= cutoff
            })
            .map_or(candidates.len(), |index| index + 1);
        candidates.truncate(keep);
        total = candidates.iter().map(|(weight, _)| weight).sum::<f32>();
    }

    let mut target = rng.next_f32() * total;
    for (weight, index) in &candidates {
        if target < *weight {
            return u32::try_from(*index)
                .map_err(|_| EngineError::Other("token id overflow".into()));
        }
        target -= *weight;
    }
    candidates
        .last()
        .and_then(|(_, index)| u32::try_from(*index).ok())
        .ok_or_else(|| EngineError::Other("empty sampling candidates".into()))
}

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub(crate) fn clock_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0x9e37_79b9_7f4a_7c15, |duration| {
            duration.as_nanos() as u64 ^ 0x9e37_79b9_7f4a_7c15
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_validation_rejects_out_of_range() {
        let mut sampling = Sampling::default();
        assert!(sampling.validate().is_ok());
        sampling.temperature = 2.1;
        assert!(matches!(
            sampling.validate(),
            Err(EngineError::InvalidSampling(_))
        ));
        sampling = Sampling {
            temperature: 1.0,
            top_p: 0.0,
            ..Sampling::default()
        };
        assert!(matches!(
            sampling.validate(),
            Err(EngineError::InvalidSampling(_))
        ));
        sampling = Sampling {
            frequency_penalty: 2.1,
            ..Sampling::default()
        };
        assert!(matches!(
            sampling.validate(),
            Err(EngineError::InvalidSampling(_))
        ));
    }

    #[test]
    fn sample_keeps_only_top_k_candidates() {
        let mut rng = Rng::new(42);
        let sampling = Sampling {
            temperature: 1.0,
            top_k: 2,
            ..Sampling::default()
        };
        let row = [0.0, 5.0, 4.0, -10.0];
        for _ in 0..32 {
            assert!(matches!(sample(&row, &sampling, &mut rng, &[]), Ok(1 | 2)));
        }
    }
}
