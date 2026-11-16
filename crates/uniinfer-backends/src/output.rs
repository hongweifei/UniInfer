//! Modality-neutral model outputs and streaming events.

use uniinfer_core::Modality;

/// Final output of one model execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelOutput {
    Text(String),
    Audio(AudioOutput),
    Image(ImageOutput),
    Video(VideoOutput),
    /// Multiple outputs produced by one execution.
    Parts(Vec<ModelOutput>),
}

impl ModelOutput {
    /// Distinct modalities carried by this output, in first-seen order.
    #[must_use]
    pub fn modalities(&self) -> Vec<Modality> {
        let mut modalities = Vec::new();
        self.collect_modalities(&mut modalities);
        modalities
    }

    fn collect_modalities(&self, modalities: &mut Vec<Modality>) {
        match self {
            Self::Text(_) => push_once(modalities, Modality::Text),
            Self::Audio(_) => push_once(modalities, Modality::Audio),
            Self::Image(_) => push_once(modalities, Modality::Image),
            Self::Video(_) => push_once(modalities, Modality::Video),
            Self::Parts(parts) => {
                for part in parts {
                    part.collect_modalities(modalities);
                }
            }
        }
    }
}

fn push_once(modalities: &mut Vec<Modality>, modality: Modality) {
    if !modalities.contains(&modality) {
        modalities.push(modality);
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
    /// A complete non-text output or multi-part event.
    Output(ModelOutput),
    Progress {
        step: usize,
        total: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_output_reports_distinct_modalities() {
        let output = ModelOutput::Parts(vec![
            ModelOutput::Text("hello".into()),
            ModelOutput::Image(ImageOutput {
                mime: "image/png".into(),
                bytes: vec![1],
                width: 1,
                height: 1,
            }),
            ModelOutput::Text("again".into()),
        ]);
        assert_eq!(output.modalities(), vec![Modality::Text, Modality::Image]);
    }
}
