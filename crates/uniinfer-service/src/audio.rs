//! Speech-model slot: one loaded session behind a single transcription permit.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_core::{
    EngineError, EngineSession, ModelManifest, PlacementPlan, Result, Tensor, TensorData,
};
use uniinfer_registry::SessionLease;

use crate::ModelView;

/// A loaded speech model plus its request permit.
pub struct AudioSlot {
    lease: Arc<SessionLease>,
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
}

impl AudioSlot {
    pub(crate) fn new(lease: Arc<SessionLease>) -> Self {
        Self {
            lease,
            permit: Arc::new(Semaphore::new(1)),
        }
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
        let input = Tensor::new("audio", vec![samples.len()], TensorData::F32(samples))?;
        let outputs = self.session().run(vec![input])?;
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
