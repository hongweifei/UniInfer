//! Runtime assembly shared by the CLI, HTTP API, benchmarks and desktop shell.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{
    ByteTokenizer, ChatMessage, ChatTemplate, GenerationOutput, GenerationSpec,
    HuggingFaceTokenizer, ModelEvent, ModelOutput, TextGenerator, Tokenizer, fallback_prompt,
};
use uniinfer_core::{
    Device, EngineError, EngineSession, ModelManifest, NativeFinish, NativeTextRequest,
    PlacementPlan, Result,
};
use uniinfer_registry::{EngineRegistry, EngineRoute, SessionLease};

mod audio;
mod image;
pub use audio::AudioSlot;
pub use image::ImageSlot;
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

/// One request executed against a loaded model.
pub enum ModelRequest {
    Text {
        prompt: String,
        images: Vec<Vec<u8>>,
        spec: GenerationSpec,
    },
    Audio {
        samples: Vec<f32>,
        deadline: Option<std::time::Instant>,
    },
    Image {
        spec: ImageGenerationSpec,
    },
}

/// Image generation parameters after manifest defaults are applied by the API.
#[derive(Debug, Clone)]
pub struct ImageGenerationSpec {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f32,
    pub seed: Option<u64>,
}

/// Common execution contract for every loaded model slot.
pub trait ModelExecutor: ModelView {
    fn execute(
        &self,
        request: ModelRequest,
        cancel: &CancelToken,
        on_event: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput>;
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
        Self::new_with_process(false)
    }

    /// Assemble engines, optionally isolating text models in worker processes.
    pub fn new_with_process(isolate_models: bool) -> Result<Self> {
        Ok(Self {
            registry: uniinfer_engine_set::build_with_process(isolate_models)?,
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
        self.load_audio_cancellable(manifest_path, CancelToken::new())
    }

    /// Load an audio model with cooperative cancellation.
    pub fn load_audio_cancellable(
        &self,
        manifest_path: &Path,
        cancel: CancelToken,
    ) -> Result<Arc<AudioSlot>> {
        let lease = Arc::new(self.registry.load_with_cancel(manifest_path, cancel)?);
        Ok(Arc::new(AudioSlot::new(lease)?))
    }

    /// Load an in-memory audio manifest against a model directory.
    pub fn load_audio_manifest(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
    ) -> Result<Arc<AudioSlot>> {
        self.load_audio_manifest_cancellable(manifest, base_dir, CancelToken::new())
    }

    pub fn load_audio_manifest_cancellable(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
        cancel: CancelToken,
    ) -> Result<Arc<AudioSlot>> {
        if !manifest
            .input_modalities()
            .contains(&uniinfer_core::Modality::Audio)
        {
            return Err(EngineError::ManifestInvalid(
                "audio runtime requires the audio modality".into(),
            ));
        }
        let lease = Arc::new(
            self.registry
                .load_manifest_with_cancel(manifest, base_dir, cancel)?,
        );
        Ok(Arc::new(AudioSlot::new(lease)?))
    }

    /// Load an image-generation model.
    pub fn load_image_manifest(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
    ) -> Result<Arc<ImageSlot>> {
        self.load_image_manifest_cancellable(manifest, base_dir, CancelToken::new())
    }

    pub fn load_image_manifest_cancellable(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
        cancel: CancelToken,
    ) -> Result<Arc<ImageSlot>> {
        if manifest.image.is_none() {
            return Err(EngineError::ManifestInvalid(
                "image runtime requires [image]".into(),
            ));
        }
        let lease = Arc::new(
            self.registry
                .load_manifest_with_cancel(manifest, base_dir, cancel)?,
        );
        Ok(Arc::new(ImageSlot::new(lease, base_dir)?))
    }

    /// Load one model and validate the text-generation contract.
    pub fn load(&self, manifest_path: &Path) -> Result<Arc<ModelSlot>> {
        self.load_cancellable(manifest_path, CancelToken::new())
    }

    /// Load a text model with cooperative cancellation.
    pub fn load_cancellable(
        &self,
        manifest_path: &Path,
        cancel: CancelToken,
    ) -> Result<Arc<ModelSlot>> {
        let lease = Arc::new(self.registry.load_with_cancel(manifest_path, cancel)?);
        let base_dir = manifest_path.parent().map_or_else(
            || std::path::PathBuf::from("."),
            std::path::Path::to_path_buf,
        );
        self.build_text_slot(lease, &base_dir)
    }

    /// Load an in-memory text manifest against a model directory.
    pub fn load_manifest(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
    ) -> Result<Arc<ModelSlot>> {
        self.load_manifest_cancellable(manifest, base_dir, CancelToken::new())
    }

    pub fn load_manifest_cancellable(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
        cancel: CancelToken,
    ) -> Result<Arc<ModelSlot>> {
        // Check the manifest before loading a session so non-text manifests fail
        // without allocating model weights.
        if manifest.text.is_none() {
            return Err(EngineError::ManifestInvalid(
                "runtime currently supports text models; manifest has no [text] section".into(),
            ));
        }
        let lease = Arc::new(
            self.registry
                .load_manifest_with_cancel(manifest, base_dir, cancel)?,
        );
        self.build_text_slot(lease, base_dir)
    }

    fn build_text_slot(
        &self,
        lease: std::sync::Arc<SessionLease>,
        base_dir: &Path,
    ) -> Result<Arc<ModelSlot>> {
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
        self.render_chat_with_tools(messages, &[], None)
    }

    /// Render chat messages with OpenAI-style tools.
    pub fn render_chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        tool_choice: Option<&serde_json::Value>,
    ) -> Result<String> {
        match &self.chat_template {
            Some(template) => template.render_with_tools(messages, tools, tool_choice),
            None => fallback_prompt(messages),
        }
    }

    /// Whether the engine decodes tokens internally.
    #[must_use]
    pub fn supports_native_generation(&self) -> bool {
        self.session().native_decoder().is_some()
    }

    /// Whether the engine session accepts an image tensor alongside token ids.
    #[must_use]
    pub fn supports_image_prompt(&self) -> bool {
        let inputs = self.session().inputs();
        inputs
            .iter()
            .any(|spec| spec.name == "image" && spec.dtype == uniinfer_core::TensorDtype::U8)
            && inputs.iter().any(|spec| {
                spec.name == "input_ids" && spec.dtype == uniinfer_core::TensorDtype::I64
            })
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
        self.generate_with(prompt, cancel, GenerationSpec::greedy(max_tokens), on_bytes)
    }

    /// Run one generation pass with explicit sampling controls.
    pub fn generate_with(
        &self,
        prompt: &str,
        cancel: &CancelToken,
        spec: GenerationSpec,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<GenerationOutput> {
        self.generate_with_images(prompt, &[], cancel, spec, on_bytes)
    }

    /// Run one generation pass with an optional image.
    pub fn generate_with_images(
        &self,
        prompt: &str,
        images: &[Vec<u8>],
        cancel: &CancelToken,
        spec: GenerationSpec,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<GenerationOutput> {
        spec.sampling.validate()?;
        if let Some(output) = self
            .session()
            .native_decoder()
            .map(|decoder| {
                decoder.generate_text(
                    NativeTextRequest {
                        prompt,
                        images,
                        max_tokens: spec.max_tokens,
                        temperature: spec.sampling.temperature,
                        top_p: spec.sampling.top_p,
                        top_k: spec.sampling.top_k,
                        seed: spec.sampling.seed,
                        frequency_penalty: spec.sampling.frequency_penalty,
                        presence_penalty: spec.sampling.presence_penalty,
                        deadline: spec.deadline,
                        cancel,
                    },
                    on_bytes,
                )
            })
            .transpose()?
        {
            return Ok(GenerationOutput {
                bytes: output.bytes,
                tokens: output.tokens,
                finish: match output.finish {
                    NativeFinish::Eos => uniinfer_backends::FinishReason::Eos,
                    NativeFinish::Length => uniinfer_backends::FinishReason::Length,
                    NativeFinish::SinkClosed => uniinfer_backends::FinishReason::SinkClosed,
                },
            });
        }
        self.generator
            .generate_with_images(self.session(), prompt, images, cancel, spec, on_bytes)
    }
}

impl ModelExecutor for ModelSlot {
    fn execute(
        &self,
        request: ModelRequest,
        cancel: &CancelToken,
        on_event: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput> {
        let ModelRequest::Text {
            prompt,
            images,
            spec,
        } = request
        else {
            return Err(EngineError::UnsupportedModality {
                format: self.manifest().format,
                modality: uniinfer_core::Modality::Audio,
            });
        };
        let mut decoder = uniinfer_backends::utf8::Utf8Decoder::new();
        let output = self.generate_with_images(&prompt, &images, cancel, spec, &mut |chunk| {
            for byte in chunk {
                if let Some(text) = decoder.push(*byte) {
                    on_event(ModelEvent::TextDelta(text));
                }
            }
            true
        })?;
        let tail = decoder.finish();
        if !tail.is_empty() {
            on_event(ModelEvent::TextDelta(tail));
        }
        Ok(ModelOutput::Text(
            String::from_utf8_lossy(&output.bytes).into_owned(),
        ))
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
    fn executor_returns_text_output_and_events() {
        let (_runtime, slot) = slot();
        let mut events = Vec::new();
        let output = slot
            .execute(
                ModelRequest::Text {
                    prompt: "ab".into(),
                    images: Vec::new(),
                    spec: GenerationSpec::greedy(3),
                },
                &CancelToken::new(),
                &mut |event| events.push(event),
            )
            .expect("execute");
        assert_eq!(output, ModelOutput::Text("cde".into()));
        assert!(!events.is_empty());
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
