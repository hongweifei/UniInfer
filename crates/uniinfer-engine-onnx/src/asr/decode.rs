//! Shared token decoding primitives for ONNX speech adapters.

pub(crate) fn argmax(values: &[f32]) -> u32 {
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

/// Collapse frame-level CTC argmax ids into emitted token ids.
pub(crate) fn ctc_tokens(logits: &[f32], frames: usize, vocab: usize, blank: u32) -> Vec<u32> {
    if vocab == 0 {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    let mut previous = None;
    for frame in 0..frames {
        let row = &logits[frame * vocab..(frame + 1) * vocab];
        let token = argmax(row);
        if token != blank && Some(token) != previous {
            tokens.push(token);
        }
        previous = Some(token);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_repeats_and_blanks() {
        let logits = vec![
            10.0, 0.0, 0.0, // a
            10.0, 0.0, 0.0, // a
            0.0, 10.0, 0.0, // blank
            0.0, 0.0, 10.0, // b
            0.0, 10.0, 0.0, // blank
        ];
        assert_eq!(ctc_tokens(&logits, 5, 3, 1), vec![0, 2]);
    }
}
