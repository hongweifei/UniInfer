//! Modality-neutral model outputs and streaming events.

use uniinfer_core::Modality;

/// Final output of one model execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelOutput {
    Text(String),
    Audio(AudioOutput),
    Image(ImageOutput),
    Video(VideoOutput),
}

impl ModelOutput {
    /// Modality carried by this output.
    #[must_use]
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Audio(_) => Modality::Audio,
            Self::Image(_) => Modality::Image,
            Self::Video(_) => Modality::Video,
        }
    }
}

/// PCM audio produced by a speech model.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOutput {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

/// Encoded image bytes plus enough metadata for an HTTP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageOutput {
    pub mime: String,
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Encoded video bytes plus basic stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoOutput {
    pub mime: String,
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub duration_ms: u64,
}

/// Incremental event emitted while a model executes.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelEvent {
    TextDelta(String),
    Progress { step: usize, total: usize },
}
