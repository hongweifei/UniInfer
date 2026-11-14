//! Runtime assembly shared by the CLI, HTTP API, benchmarks and desktop shell.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{ByteTokenizer, HuggingFaceTokenizer, TextGenerator, Tokenizer};
use uniinfer_core::{Device, EngineError, EngineSession, ModelManifest, PlacementPlan, Result};
use uniinfer_engine_set;
use uniinfer_registry::{EngineRegistry, SessionLease};

pub use uniinfer_core::CancelToken;

/// Owns the engine registry and the device inventory.
pub struct Runtime {
    registry: EngineRegistry,
}

impl Runtime {
    /// Assemble the default engine set and probe its devices.
    pub fn new() -> Result<Self> {
        Ok(Self {
            registry: uniinfer_engine_set::build()?,
        })
    }

    #[must_use]
    pub fn devices(&self) -> &[Device] {
        self.registry.inventory().devices()
    }

    #[must_use]
    pub fn active_sessions(&self) -> usize {
        self.registry.active_sessions()
    }

    /// Load one model and validate the text-generation contract.
    pub fn load(&self, manifest_path: &Path) -> Result<Arc<ModelSlot>> {
        // Backend dispatch is still text-only. Check before loading a session
        // so non-text manifests fail without allocating model weights.
        let manifest = ModelManifest::load(manifest_path)?;
        let text = manifest.text.as_ref().ok_or_else(|| {
            EngineError::ManifestInvalid(
                "runtime currently supports text models; manifest has no [text] section".into(),
            )
        })?;
        let lease = Arc::new(self.registry.load(manifest_path)?);
        let manifest = lease.manifest();
        let tokenizer = manifest.tokenizer.as_deref().unwrap_or_default();
        let tokenizer: Arc<dyn Tokenizer> = match tokenizer {
            "bytes" => Arc::new(ByteTokenizer::new(text.vocab_size)?),
            "hf" => {
                let file = text.tokenizer_file.as_ref().ok_or_else(|| {
                    EngineError::ManifestInvalid("text.tokenizer_file is required for hf".into())
                })?;
                let base = manifest_path.parent().map_or_else(
                    || std::path::PathBuf::from("."),
                    std::path::Path::to_path_buf,
                );
                Arc::new(HuggingFaceTokenizer::from_file(base.join(file))?)
            }
            other => {
                return Err(EngineError::UnsupportedEngine(format!(
                    "tokenizer {other:?}"
                )));
            }
        };
        let generator = TextGenerator::new(
            tokenizer,
            text.vocab_size,
            manifest.context_len,
            text.eos_token,
        );

        Ok(Arc::new(ModelSlot {
            generator,
            lease,
            permit: Arc::new(Semaphore::new(1)),
        }))
    }
}

/// A loaded model with a single generation permit.
pub struct ModelSlot {
    lease: Arc<SessionLease>,
    generator: TextGenerator,
    permit: Arc<Semaphore>,
}

impl ModelSlot {
    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        self.lease.manifest()
    }

    #[must_use]
    pub fn placement(&self) -> &PlacementPlan {
        self.lease.placement()
    }

    #[must_use]
    pub fn engine_name(&self) -> &str {
        self.lease.engine_name()
    }

    #[must_use]
    pub fn session(&self) -> &dyn EngineSession {
        self.lease.session()
    }

    /// Acquire the generation permit, returning `Busy` on timeout.
    pub async fn acquire(&self, wait: Duration) -> Result<OwnedSemaphorePermit> {
        match tokio::time::timeout(wait, Arc::clone(&self.permit).acquire_owned()).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(EngineError::Other("model semaphore closed".into())),
            Err(_) => Err(EngineError::Busy),
        }
    }

    /// Try to acquire the generation permit without waiting.
    #[must_use]
    pub fn try_acquire(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.permit).try_acquire_owned().ok()
    }

    /// Count prompt tokens with the slot's tokenizer.
    pub fn count_tokens(&self, prompt: &str) -> Result<usize> {
        self.generator.count_tokens(prompt)
    }

    /// Run one greedy byte generation pass.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &CancelToken,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<Vec<u8>> {
        self.generator
            .generate(self.session(), prompt, max_tokens, cancel, on_bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use super::*;

    fn slot() -> (Runtime, Arc<ModelSlot>) {
        let runtime = Runtime::new().expect("runtime");
        let manifest =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
        let slot = runtime.load(&manifest).expect("slot");
        (runtime, slot)
    }

    #[tokio::test]
    async fn second_acquire_times_out_as_busy() {
        let (_runtime, slot) = slot();
        let first = slot
            .acquire(Duration::from_millis(50))
            .await
            .expect("first");
        let error = slot
            .acquire(Duration::from_millis(10))
            .await
            .expect_err("must time out");
        assert!(matches!(error, EngineError::Busy));
        drop(first);
    }

    #[test]
    fn generate_uses_manifest_context() {
        let (_runtime, slot) = slot();
        let bytes = slot
            .generate("ab", 3, &CancelToken::new(), &mut |_| true)
            .expect("generate");
        assert_eq!(bytes, b"cde");
    }
}
