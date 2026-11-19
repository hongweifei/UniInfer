//! Tokenizer contract and built-in implementations.

use uniinfer_core::{EngineError, Result};

/// Tokenizer contract used by the text generator.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, token: u32) -> Result<Vec<u8>>;
    fn vocab_size(&self) -> usize;
    /// Padding token when the tokenizer owns a fixed convention (CLIP).
    fn padding_token(&self) -> Option<u32> {
        None
    }
}

/// Byte tokenizer used by the tiny fixture.
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
    clip_special: Option<(u32, u32)>,
}

impl HuggingFaceTokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| EngineError::ManifestInvalid(format!("tokenizer file: {error}")))?;
        Ok(Self {
            inner,
            clip_special: None,
        })
    }

    /// Build a CLIP-style BPE tokenizer from the vocab/merges files that some
    /// diffusion repositories ship instead of a `tokenizer.json`.
    pub fn from_vocab_merges(
        vocab_path: impl AsRef<std::path::Path>,
        merges_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        let model = BPE::builder()
            .files(
                vocab_path.as_ref().to_string_lossy().into_owned(),
                merges_path.as_ref().to_string_lossy().into_owned(),
            )
            .end_of_word_suffix("</w>".to_owned())
            .build()
            .map_err(|error| EngineError::ManifestInvalid(format!("tokenizer bpe: {error}")))?;
        let mut inner = tokenizers::Tokenizer::new(model);
        let mut pre_tokenizer = ByteLevel::default();
        pre_tokenizer.add_prefix_space = false;
        inner.with_pre_tokenizer(Some(pre_tokenizer));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        let special = match (
            inner.token_to_id("<|startoftext|>"),
            inner.token_to_id("<|endoftext|>"),
        ) {
            (Some(start), Some(end)) => Some((start, end)),
            _ => None,
        };
        Ok(Self {
            inner,
            clip_special: special,
        })
    }
}

impl Tokenizer for HuggingFaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self
            .inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| EngineError::Other(format!("tokenize: {error}")))?;
        if let Some((start, end)) = self.clip_special {
            ids.insert(0, start);
            ids.push(end);
        }
        Ok(ids)
    }

    fn padding_token(&self) -> Option<u32> {
        self.clip_special.map(|(_, end)| end)
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
