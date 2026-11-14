//! Runtime assembly shared by the CLI, HTTP API, benchmarks and desktop shell.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{ByteTokenizer, TextGenerator};
use uniinfer_core::{Device, EngineError, EngineSession, ModelManifest, PlacementPlan, Result};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;
use uniinfer_registry::{EngineRegistry, SessionLease};

pub use uniinfer_core::CancelToken;

/// Owns the engine registry and the device inventory.
pub struct Runtime {
    registry: EngineRegistry,
}

impl Runtime {
    /// Assemble the default engine set and probe its devices.
    pub fn new() -> Result<Self> {
        let engine = OnnxEngine::new()?;
        let mut inventory = DeviceInventory::probe();
        for device in engine.probe_devices() {
            inventory.register(device);
        }
        let mut registry = EngineRegistry::new(inventory);
        registry.register(Arc::new(engine));
        Ok(Self { registry })
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
        let lease = Arc::new(self.registry.load(manifest_path)?);
        let manifest = lease.manifest();
        let tokenizer = manifest.tokenizer.as_deref().unwrap_or_default();
        if tokenizer != "bytes" {
            return Err(EngineError::UnsupportedEngine(format!(
                "tokenizer {tokenizer:?}; byte tokenizer is required"
            )));
        }
        let text = manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("manifest has no [text] section".into()))?;
        let generator = TextGenerator::new(
            Arc::new(ByteTokenizer::new(text.vocab_size)?),
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
