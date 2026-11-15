//! Runtime assembly shared by the CLI, HTTP API, benchmarks and desktop shell.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{
    ByteTokenizer, ChatMessage, ChatTemplate, GenerationOutput, HuggingFaceTokenizer, Sampling,
    TextGenerator, Tokenizer, fallback_prompt,
};
use uniinfer_core::{Device, EngineError, EngineSession, ModelManifest, PlacementPlan, Result};
use uniinfer_registry::{EngineRegistry, EngineRoute, SessionLease};

mod audio;
pub use audio::AudioSlot;
pub use uniinfer_core::CancelToken;

/// Acquire one of a slot's request permits, returning `Busy` on timeout.
pub(crate) async fn acquire_permit(
    permit: &Arc<Semaphore>,
    wait: Duration,
) -> Result<OwnedSemaphorePermit> {
    match tokio::time::timeout(wait, Arc::clone(permit).acquire_owned()).await {
        Ok(Ok(permit)) => Ok(permit),
        Ok(Err(_)) => Err(EngineError::Other("slot semaphore closed".into())),
        Err(_) => Err(EngineError::Busy),
    }
}

/// Read-only view shared by text and speech slots.
pub trait ModelView: Send + Sync {
    fn manifest(&self) -> &ModelManifest;
    fn placement(&self) -> &PlacementPlan;
    fn engine_name(&self) -> &str;
}

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

    /// Resolve engine and device placement without loading weights.
    pub fn probe_manifest(&self, manifest: &ModelManifest) -> Result<EngineRoute> {
        self.registry.route(manifest)
    }

    /// Load a manifest and resolve its engine route without loading weights.
    pub fn probe(&self, manifest_path: &Path) -> Result<EngineRoute> {
        let manifest = ModelManifest::load(manifest_path)?;
        self.probe_manifest(&manifest)
    }

    /// Load an audio model for transcription.
    pub fn load_audio(&self, manifest_path: &Path) -> Result<Arc<AudioSlot>> {
        let manifest = ModelManifest::load(manifest_path)?;
        let base_dir = ModelManifest::base_dir(manifest_path);
        self.load_audio_manifest(manifest, &base_dir)
    }

    /// Load an in-memory audio manifest against a model directory.
    pub fn load_audio_manifest(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
    ) -> Result<Arc<AudioSlot>> {
        if !manifest
            .modalities
            .contains(&uniinfer_core::Modality::Audio)
        {
            return Err(EngineError::ManifestInvalid(
                "audio runtime requires the audio modality".into(),
            ));
        }
        let lease = Arc::new(self.registry.load_manifest(manifest, base_dir)?);
        Ok(Arc::new(AudioSlot::new(lease)?))
    }

    /// Load one model and validate the text-generation contract.
    pub fn load(&self, manifest_path: &Path) -> Result<Arc<ModelSlot>> {
        let manifest = ModelManifest::load(manifest_path)?;
        let base_dir = ModelManifest::base_dir(manifest_path);
        self.load_manifest(manifest, &base_dir)
    }

    /// Load an in-memory text manifest against a model directory.
    pub fn load_manifest(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
    ) -> Result<Arc<ModelSlot>> {
        // Backend dispatch is still text-only. Check before loading a session
        // so non-text manifests fail without allocating model weights.
        let text = manifest.text.clone().ok_or_else(|| {
            EngineError::ManifestInvalid(
                "runtime currently supports text models; manifest has no [text] section".into(),
            )
        })?;
        let lease = Arc::new(self.registry.load_manifest(manifest, base_dir)?);
        let manifest = lease.manifest();
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
}

/// A loaded model with a single generation permit.
pub struct ModelSlot {
    lease: Arc<SessionLease>,
    generator: TextGenerator,
    chat_template: Option<ChatTemplate>,
    permit: Arc<Semaphore>,
}

impl ModelView for ModelSlot {
    fn manifest(&self) -> &ModelManifest {
        self.lease.manifest()
    }

    fn placement(&self) -> &PlacementPlan {
        self.lease.placement()
    }

    fn engine_name(&self) -> &str {
        self.lease.engine_name()
    }
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
        acquire_permit(&self.permit, wait).await
    }

    /// Try to acquire the generation permit without waiting.
    #[must_use]
    pub fn try_acquire(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.permit).try_acquire_owned().ok()
    }

    /// Render chat messages with the model template, or the fallback prompt.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        match &self.chat_template {
            Some(template) => template.render(messages),
            None => fallback_prompt(messages),
        }
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
    ) -> Result<GenerationOutput> {
        self.generate_with(prompt, max_tokens, cancel, &Sampling::greedy(), on_bytes)
    }

    /// Run one generation pass with explicit sampling controls.
    pub fn generate_with(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &CancelToken,
        sampling: &Sampling,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<GenerationOutput> {
        self.generator.generate_with(
            self.session(),
            prompt,
            max_tokens,
            cancel,
            sampling,
            on_bytes,
        )
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
        let output = slot
            .generate("ab", 3, &CancelToken::new(), &mut |_| true)
            .expect("generate");
        assert_eq!(output.bytes, b"cde");
        assert_eq!(output.tokens, 3);
    }
}
