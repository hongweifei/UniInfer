//! Runtime assembly shared by the CLI, HTTP API, benchmarks and desktop shell.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{GenerateOptions, generate_bytes_cancellable};
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

        Ok(Arc::new(ModelSlot {
            context_len: manifest.context_len,
            vocab: text.vocab_size,
            eos_token: text.eos_token,
            lease,
            permit: Arc::new(Semaphore::new(1)),
        }))
    }
}

/// A loaded model with a single generation permit.
pub struct ModelSlot {
    lease: Arc<SessionLease>,
    vocab: usize,
    eos_token: Option<u32>,
    context_len: usize,
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

    /// Run one greedy byte generation pass.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &CancelToken,
        on_byte: &mut dyn FnMut(u8) -> bool,
    ) -> Result<Vec<u8>> {
        let options = GenerateOptions {
            max_tokens,
            eos_token: self.eos_token,
        };
        generate_bytes_cancellable(
            self.session(),
            prompt,
            self.vocab,
            self.context_len,
            &options,
            cancel,
            on_byte,
        )
    }
}
