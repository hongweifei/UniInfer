//! Tokenizer contract and built-in implementations.

use uniinfer_core::{EngineError, Result};

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
