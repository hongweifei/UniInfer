//! Build a text model slot from a loaded session lease.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Semaphore;
use uniinfer_backends::{
    ByteTokenizer, ChatTemplate, HuggingFaceTokenizer, TextGenerator, Tokenizer,
};
use uniinfer_core::{EngineError, Result};
use uniinfer_registry::SessionLease;

use crate::ModelSlot;

pub(crate) fn build(lease: Arc<SessionLease>, base_dir: &Path) -> Result<Arc<ModelSlot>> {
    let manifest = lease.manifest();
    let text = manifest
        .text
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("manifest has no [text] section".into()))?;
    let tokenizer = manifest.tokenizer.as_deref().unwrap_or_default();
    let base = base_dir.to_path_buf();
    let tokenizer: Arc<dyn Tokenizer> = match tokenizer {
        "bytes" => Arc::new(ByteTokenizer::new(text.vocab_size)?),
        "hf" => {
            let file = text.tokenizer_file.as_ref().ok_or_else(|| {
                EngineError::ManifestInvalid("text.tokenizer_file is required for hf".into())
            })?;
            Arc::new(HuggingFaceTokenizer::from_file(base.join(file))?)
        }
        other => {
            return Err(EngineError::UnsupportedEngine(format!(
                "tokenizer {other:?}"
            )));
        }
    };
    let chat_template = text
        .tokenizer_file
        .as_ref()
        .map(|file| {
            ChatTemplate::from_tokenizer_config(
                &base.join(file).with_file_name("tokenizer_config.json"),
            )
        })
        .transpose()?
        .flatten();
    let generator = TextGenerator::new(
        tokenizer,
        text.vocab_size,
        manifest.context_len,
        text.eos_token,
    );

    Ok(Arc::new(ModelSlot {
        generator,
        chat_template,
        lease,
        permit: Arc::new(Semaphore::new(1)),
    }))
}
