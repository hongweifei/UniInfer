//! Speech-model slot: one loaded session behind a single transcription permit.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{ModelEvent, ModelOutput};
use uniinfer_core::{
    AudioConfig, EngineError, EngineSession, ModelManifest, PlacementPlan, Residency, Result,
    Tensor, TensorData, TensorDtype,
};
use uniinfer_registry::SessionLease;

use crate::{ModelExecutor, ModelRequest, ModelView};

/// A loaded speech model plus its request permit.
pub struct AudioSlot {
    lease: Arc<SessionLease>,
    audio: AudioConfig,
    permit: Arc<Semaphore>,
}

impl ModelView for AudioSlot {
    fn manifest(&self) -> &ModelManifest {
        AudioSlot::manifest(self)
    }

    fn placement(&self) -> &PlacementPlan {
        AudioSlot::placement(self)
    }

    fn engine_name(&self) -> &str {
        AudioSlot::engine_name(self)
    }

    fn residency(&self) -> Option<Residency> {
        self.lease.residency()
    }
}

impl AudioSlot {
    pub(crate) fn new(lease: Arc<SessionLease>) -> Result<Self> {
        let audio = lease.manifest().audio.clone().unwrap_or_default();
        let session = lease.session();
        let input = session
            .inputs()
            .iter()
            .find(|spec| spec.name == audio.input && spec.dtype == TensorDtype::F32)
            .ok_or_else(|| {
                EngineError::UnsupportedEngine(format!(
                    "audio model must expose F32 input {:?}",
                    audio.input
                ))
            })?;
        let output = session
            .outputs()
            .iter()
            .find(|spec| spec.name == audio.output && spec.dtype == TensorDtype::U8)
            .ok_or_else(|| {
                EngineError::UnsupportedEngine(format!(
                    "audio model must expose U8 output {:?}",
                    audio.output
                ))
            })?;
        let _ = (input, output);
        Ok(Self {
            lease,
            audio,
            permit: Arc::new(Semaphore::new(1)),
        })
    }

    #[must_use]
    pub fn audio_config(&self) -> &AudioConfig {
        &self.audio
    }

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

    /// Acquire the transcription permit, returning `Busy` on timeout.
    pub async fn acquire(&self, wait: Duration) -> Result<OwnedSemaphorePermit> {
        crate::acquire_permit(&self.permit, wait).await
    }

    /// Transcribe 16 kHz mono F32 samples with the loaded session.
    pub fn transcribe(&self, samples: Vec<f32>) -> Result<String> {
        self.transcribe_with_deadline(samples, None)
    }

    /// Transcribe with an optional wall-clock deadline.
    pub fn transcribe_with_deadline(
        &self,
        samples: Vec<f32>,
        deadline: Option<std::time::Instant>,
    ) -> Result<String> {
        let input = Tensor::new("audio", vec![samples.len()], TensorData::F32(samples))?;
        let outputs = self.session().run_with_deadline(vec![input], deadline)?;
        let output = outputs
            .iter()
            .find(|tensor| tensor.name == "text")
            .ok_or_else(|| EngineError::Other("model has no text output".into()))?;
        match &output.data {
            TensorData::U8(bytes) => String::from_utf8(bytes.clone())
                .map_err(|error| EngineError::InvalidAudio(format!("transcript: {error}"))),
            _ => Err(EngineError::Other("text output must be u8".into())),
        }
    }
}

impl ModelExecutor for AudioSlot {
    fn execute(
        &self,
        request: ModelRequest,
        _cancel: &uniinfer_core::CancelToken,
        on_event: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput> {
        let ModelRequest::Audio { samples, deadline } = request else {
            return Err(EngineError::UnsupportedModality {
                format: self.manifest().format,
                modality: uniinfer_core::Modality::Text,
            });
        };
        let text = self.transcribe_with_deadline(samples, deadline)?;
        on_event(ModelEvent::TextDelta(text.clone()));
        Ok(ModelOutput::Text(text))
    }
}
