//! Modality backends. M0 provides byte-level text generation over any engine.

use uniinfer_core::{
    CancelToken, EngineError, EngineSession, Result, Tensor, TensorData, TensorDtype,
};

/// Greedy decoding parameters.
#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    pub eos_token: Option<u32>,
}

/// Generate bytes with greedy argmax over a `[1, seq, vocab]` logits tensor.
///
/// The prompt and generated tokens use a byte tokenizer: each id maps to one
/// byte. Engines see the full sequence on every step, so the model must accept
/// a growing `input_ids` without a KV cache.
pub fn generate_bytes(
    session: &dyn EngineSession,
    prompt: &str,
    vocab_size: usize,
    context_len: usize,
    options: &GenerateOptions,
    on_byte: &mut dyn FnMut(u8) -> bool,
) -> Result<Vec<u8>> {
    generate_bytes_cancellable(
        session,
        prompt,
        vocab_size,
        context_len,
        options,
        &CancelToken::new(),
        on_byte,
    )
}

/// Same as [`generate_bytes`] with a cooperative cancellation flag checked
/// between model steps.
#[allow(clippy::too_many_arguments)]
pub fn generate_bytes_cancellable(
    session: &dyn EngineSession,
    prompt: &str,
    vocab_size: usize,
    context_len: usize,
    options: &GenerateOptions,
    cancel: &CancelToken,
    on_byte: &mut dyn FnMut(u8) -> bool,
) -> Result<Vec<u8>> {
    if cancel.is_cancelled() {
        return Err(EngineError::Cancelled);
    }
    let inputs = session.inputs();
    let id_spec = inputs
        .iter()
        .find(|spec| spec.dtype == TensorDtype::I64 && spec.name.contains("input_ids"))
        .or_else(|| inputs.iter().find(|spec| spec.dtype == TensorDtype::I64))
        .ok_or_else(|| EngineError::Other("model has no int64 input_ids".into()))?;
    let mask_spec = inputs
        .iter()
        .find(|spec| spec.dtype == TensorDtype::I64 && spec.name.contains("attention_mask"));

    let logits_spec = session
        .outputs()
        .iter()
        .find(|spec| spec.dtype == TensorDtype::F32 && spec.name.contains("logits"))
        .or_else(|| {
            session
                .outputs()
                .iter()
                .find(|spec| spec.dtype == TensorDtype::F32)
        })
        .ok_or_else(|| EngineError::Other("model has no float logits output".into()))?;

    if vocab_size == 0 {
        return Err(EngineError::ManifestInvalid(
            "text.vocab_size is required for byte generation".into(),
        ));
    }
    if options.max_tokens == 0 {
        return Ok(Vec::new());
    }

    let prompt_bytes: &[u8] = if prompt.is_empty() {
        b" "
    } else {
        prompt.as_bytes()
    };
    let required = prompt_bytes.len() + options.max_tokens;
    if required > context_len {
        return Err(EngineError::InputTooLong {
            input: required,
            limit: context_len,
        });
    }

    let mut ids: Vec<i64> = prompt_bytes.iter().map(|byte| i64::from(*byte)).collect();
    let mut generated = Vec::with_capacity(options.max_tokens);

    for _ in 0..options.max_tokens {
        let length = ids.len();
        let mut tensors = vec![Tensor::new(
            id_spec.name.clone(),
            vec![1, length],
            TensorData::I64(ids.clone()),
        )?];
        if let Some(mask) = mask_spec {
            tensors.push(Tensor::new(
                mask.name.clone(),
                vec![1, length],
                TensorData::I64(vec![1; length]),
            )?);
        }

        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let outputs = session.run(tensors)?;
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let logits = outputs
            .iter()
            .find(|tensor| tensor.name == logits_spec.name)
            .ok_or_else(|| EngineError::Other(format!("missing output {}", logits_spec.name)))?;

        let (vocab, offset) = logits_layout(logits, vocab_size)?;
        let start = offset;
        let row = &as_f32(logits)?[start..start + vocab];
        let next = argmax(row);
        let byte = u8::try_from(next)
            .map_err(|_| EngineError::Other(format!("token id {next} does not fit a byte")))?;

        generated.push(byte);
        if !on_byte(byte) {
            break;
        }
        if options.eos_token == Some(next) {
            break;
        }
        ids.push(i64::from(next));
    }

    Ok(generated)
}

fn as_f32(tensor: &Tensor) -> Result<&[f32]> {
    match &tensor.data {
        TensorData::F32(values) => Ok(values),
        other => Err(EngineError::Other(format!(
            "expected f32 logits, got {:?}",
            other.dtype()
        ))),
    }
}

/// Return `(vocab, offset_of_last_position)` for rank-2 or rank-3 logits.
fn logits_layout(tensor: &Tensor, vocab_size: usize) -> Result<(usize, usize)> {
    match tensor.shape.as_slice() {
        [1, seq, vocab] if *vocab == vocab_size => {
            let last = (*seq - 1) * vocab;
            Ok((vocab_size, last))
        }
        [1, vocab] if *vocab == vocab_size => Ok((vocab_size, 0)),
        shape => Err(EngineError::Other(format!(
            "unexpected logits shape {shape:?}, expected [1, seq, {vocab_size}]"
        ))),
    }
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use uniinfer_core::TensorSpec;

    struct Scripted {
        inputs: Vec<TensorSpec>,
        outputs: Vec<TensorSpec>,
        seen: Mutex<Vec<Vec<i64>>>,
    }

    impl Scripted {
        fn new() -> Self {
            Self {
                inputs: vec![TensorSpec {
                    name: "input_ids".into(),
                    dtype: TensorDtype::I64,
                    shape: vec![Some(1), None],
                }],
                outputs: vec![TensorSpec {
                    name: "logits".into(),
                    dtype: TensorDtype::F32,
                    shape: vec![Some(1), None, Some(256)],
                }],
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    impl EngineSession for Scripted {
        fn inputs(&self) -> &[TensorSpec] {
            &self.inputs
        }

        fn outputs(&self) -> &[TensorSpec] {
            &self.outputs
        }

        fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
            let ids = match &inputs[0].data {
                TensorData::I64(values) => values.clone(),
                _ => return Err(EngineError::Other("expected i64".into())),
            };
            let seq = ids.len();
            self.seen.lock().expect("lock").push(ids.clone());
            let mut logits = vec![0.0f32; seq * 256];
            for position in 0..seq {
                let next = (ids[position] + 1) % 256;
                logits[position * 256 + next as usize] = 1.0;
            }
            Ok(vec![Tensor::new(
                "logits",
                vec![1, seq, 256],
                TensorData::F32(logits),
            )?])
        }

        fn cancel(&self) {}
    }

    #[test]
    fn greedy_byte_generation_advances_one_byte_per_step() {
        let session = Scripted::new();
        let mut bytes = Vec::new();
        let generated = generate_bytes(
            &session,
            "ab",
            256,
            16,
            &GenerateOptions {
                max_tokens: 3,
                eos_token: None,
            },
            &mut |byte| {
                bytes.push(byte);
                true
            },
        )
        .expect("generate");

        assert_eq!(generated, b"cde");
        assert_eq!(bytes, b"cde");
        let seen = session.seen.lock().expect("lock");
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0], vec![97, 98]);
        assert_eq!(seen[2], vec![97, 98, 99, 100]);
    }

    #[test]
    fn context_overflow_is_rejected() {
        let session = Scripted::new();
        let error = generate_bytes(
            &session,
            "abcdef",
            256,
            4,
            &GenerateOptions {
                max_tokens: 3,
                eos_token: None,
            },
            &mut |_| true,
        )
        .expect_err("must fail");
        assert!(matches!(error, EngineError::InputTooLong { .. }));
    }
}

#[cfg(test)]
mod cancel_tests {
    use super::*;

    struct Noop;

    impl EngineSession for Noop {
        fn inputs(&self) -> &[uniinfer_core::TensorSpec] {
            &[]
        }

        fn outputs(&self) -> &[uniinfer_core::TensorSpec] {
            &[]
        }

        fn run(&self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
            panic!("must not run after cancellation");
        }

        fn cancel(&self) {}
    }

    #[test]
    fn pre_cancelled_generation_returns_cancelled() {
        let token = CancelToken::new();
        token.cancel();
        let error = generate_bytes_cancellable(
            &Noop,
            "ab",
            256,
            16,
            &GenerateOptions {
                max_tokens: 3,
                eos_token: None,
            },
            &token,
            &mut |_| true,
        )
        .expect_err("must be cancelled");
        assert!(matches!(error, EngineError::Cancelled));
    }
}
