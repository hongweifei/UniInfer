//! Text generation over an abstract tokenizer.

use std::sync::Arc;

use uniinfer_core::{
    CancelToken, EngineError, EngineSession, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};

/// Tokenizer contract used by the text generator.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, token: u32) -> Result<Vec<u8>>;
    fn vocab_size(&self) -> usize;
}

/// Byte tokenizer used by the M0/M1 fixture.
#[derive(Debug, Clone)]
pub struct ByteTokenizer {
    vocab: usize,
}

impl ByteTokenizer {
    pub fn new(vocab: usize) -> Result<Self> {
        if vocab == 0 || vocab > 256 {
            return Err(EngineError::ManifestInvalid(format!(
                "byte tokenizer vocab must be within 1..=256, got {vocab}"
            )));
        }
        Ok(Self { vocab })
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = text.bytes().map(u32::from).collect::<Vec<_>>();
        if let Some(token) = tokens.iter().find(|token| **token >= self.vocab as u32) {
            return Err(EngineError::InputTooLong {
                input: *token as usize + 1,
                limit: self.vocab,
            });
        }
        Ok(tokens)
    }

    fn decode(&self, token: u32) -> Result<Vec<u8>> {
        if token >= self.vocab as u32 {
            return Err(EngineError::Other(format!(
                "token {token} outside byte vocab {}",
                self.vocab
            )));
        }
        Ok(vec![token as u8])
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// Greedy text generator bound to one tokenizer and context size.
pub struct TextGenerator {
    tokenizer: Arc<dyn Tokenizer>,
    vocab_size: usize,
    context_len: usize,
    eos_token: Option<u32>,
}

impl TextGenerator {
    #[must_use]
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        vocab_size: usize,
        context_len: usize,
        eos_token: Option<u32>,
    ) -> Self {
        Self {
            tokenizer,
            vocab_size,
            context_len,
            eos_token,
        }
    }

    #[must_use]
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Generate tokens until `max_tokens`, EOS, sink rejection or cancellation.
    pub fn generate(
        &self,
        session: &dyn EngineSession,
        prompt: &str,
        max_tokens: usize,
        cancel: &CancelToken,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<Vec<u8>> {
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if max_tokens == 0 {
            return Ok(Vec::new());
        }

        let prompt = if prompt.is_empty() { " " } else { prompt };
        let mut ids: Vec<i64> = self
            .tokenizer
            .encode(prompt)?
            .into_iter()
            .map(i64::from)
            .collect();

        let required = ids.len() + max_tokens;
        if required > self.context_len {
            return Err(EngineError::InputTooLong {
                input: required,
                limit: self.context_len,
            });
        }

        let id_spec = find_input(session, "input_ids")
            .or_else(|| find_i64_input(session))
            .ok_or_else(|| EngineError::Other("model has no int64 input_ids".into()))?;
        let mask_spec = find_input(session, "attention_mask");
        let logits_spec = find_logits(session)
            .ok_or_else(|| EngineError::Other("model has no float logits output".into()))?;

        let mut generated = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }

            let length = ids.len();
            let mut tensors = vec![Tensor::new(
                id_spec.name.clone(),
                vec![1, length],
                TensorData::I64(ids.clone()),
            )?];
            if let Some(mask) = &mask_spec {
                tensors.push(Tensor::new(
                    mask.name.clone(),
                    vec![1, length],
                    TensorData::I64(vec![1; length]),
                )?);
            }

            let outputs = session.run(tensors)?;
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }

            let logits = outputs
                .iter()
                .find(|tensor| tensor.name == logits_spec.name)
                .ok_or_else(|| {
                    EngineError::Other(format!("missing output {}", logits_spec.name))
                })?;
            let next = argmax(logits, self.vocab_size, self.tokenizer.vocab_size())?;
            let bytes = self.tokenizer.decode(next)?;

            generated.extend_from_slice(&bytes);
            if !on_bytes(&bytes) {
                break;
            }
            if self.eos_token == Some(next) {
                break;
            }
            ids.push(i64::from(next));
        }
        Ok(generated)
    }
}

fn find_input(session: &dyn EngineSession, name: &str) -> Option<TensorSpec> {
    session
        .inputs()
        .iter()
        .find(|spec| spec.dtype == TensorDtype::I64 && spec.name.contains(name))
        .cloned()
}

fn find_i64_input(session: &dyn EngineSession) -> Option<TensorSpec> {
    session
        .inputs()
        .iter()
        .find(|spec| spec.dtype == TensorDtype::I64)
        .cloned()
}

fn find_logits(session: &dyn EngineSession) -> Option<TensorSpec> {
    session
        .outputs()
        .iter()
        .find(|spec| spec.dtype == TensorDtype::F32 && spec.name.contains("logits"))
        .or_else(|| {
            session
                .outputs()
                .iter()
                .find(|spec| spec.dtype == TensorDtype::F32)
        })
        .cloned()
}

fn argmax(tensor: &Tensor, vocab_size: usize, tokenizer_vocab: usize) -> Result<u32> {
    let values = match &tensor.data {
        TensorData::F32(values) => values,
        other => {
            return Err(EngineError::Other(format!(
                "expected f32 logits, got {:?}",
                other.dtype()
            )));
        }
    };
    let (vocab, offset) = match tensor.shape.as_slice() {
        [1, seq, vocab] if *vocab == vocab_size => (vocab_size, (*seq - 1) * vocab_size),
        [1, vocab] if *vocab == vocab_size => (vocab_size, 0),
        shape => {
            return Err(EngineError::Other(format!(
                "unexpected logits shape {shape:?}, expected [1, seq, {vocab_size}]"
            )));
        }
    };
    let row = &values[offset..offset + vocab.min(tokenizer_vocab)];
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in row.iter().enumerate() {
        if value.is_finite() && *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    u32::try_from(best).map_err(|_| EngineError::Other("token id overflow".into()))
}

pub struct HuggingFaceTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HuggingFaceTokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| EngineError::ManifestInvalid(format!("tokenizer file: {error}")))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| EngineError::Other(format!("tokenize: {error}")))
    }

    fn decode(&self, token: u32) -> Result<Vec<u8>> {
        self.inner
            .decode(&[token], false)
            .map(String::into_bytes)
            .map_err(|error| EngineError::Other(format!("detokenize: {error}")))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

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

    fn generator() -> TextGenerator {
        TextGenerator::new(
            Arc::new(ByteTokenizer::new(256).expect("tokenizer")),
            256,
            64,
            None,
        )
    }

    #[test]
    fn greedy_byte_generation_advances_one_byte_per_step() {
        let session = Scripted::new();
        let mut bytes = Vec::new();
        let generated = generator()
            .generate(&session, "ab", 3, &CancelToken::new(), &mut |chunk| {
                bytes.extend_from_slice(chunk);
                true
            })
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
        let generator = TextGenerator::new(
            Arc::new(ByteTokenizer::new(256).expect("tokenizer")),
            256,
            8,
            None,
        );
        let error = generator
            .generate(&session, "abcdef", 3, &CancelToken::new(), &mut |_| true)
            .expect_err("must fail");
        assert!(matches!(error, EngineError::InputTooLong { .. }));
    }

    #[test]
    fn pre_cancelled_generation_returns_cancelled() {
        let session = Scripted::new();
        let cancel = CancelToken::new();
        cancel.cancel();
        let error = generator()
            .generate(&session, "ab", 3, &cancel, &mut |_| true)
            .expect_err("must be cancelled");
        assert!(matches!(error, EngineError::Cancelled));
    }
}
