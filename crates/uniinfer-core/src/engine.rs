use std::path::PathBuf;

use crate::device::{DeviceBudget, DeviceClass, PlacementPlan};
use crate::error::Result;
use crate::manifest::{Modality, ModelFormat, ModelManifest, Quantization};
use crate::tensor::{Tensor, TensorSpec};

/// Static capability matrix reported by an engine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineCapabilities {
    pub formats: Vec<ModelFormat>,
    pub modalities: Vec<(ModelFormat, Vec<Modality>)>,
    pub quantizations: Vec<(ModelFormat, Vec<Quantization>)>,
    pub devices: Vec<DeviceClass>,
}

/// Everything an engine needs to load one model.
#[derive(Debug, Clone)]
pub struct EngineLoadSpec {
    pub manifest: ModelManifest,
    pub base_dir: PathBuf,
    pub placement: PlacementPlan,
    pub budget: DeviceBudget,
}

/// A loaded model session.
pub trait EngineSession: Send + Sync {
    fn inputs(&self) -> &[TensorSpec];
    fn outputs(&self) -> &[TensorSpec];

    /// Run one forward pass.
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;

    /// Streaming entry point. The default yields the full result as one chunk;
    /// engines that can emit partial results override it.
    fn run_streaming(
        &self,
        inputs: Vec<Tensor>,
        sink: &mut dyn FnMut(Vec<Tensor>) -> bool,
    ) -> Result<()> {
        let outputs = self.run(inputs)?;
        sink(outputs);
        Ok(())
    }

    /// Idempotent cancellation. Pending and subsequent runs return `Cancelled`.
    fn cancel(&self);
}

/// Engine contract.
pub trait InferenceEngine: Send + Sync {
    fn engine_name(&self) -> &'static str;

    fn supported_formats(&self) -> &[ModelFormat];

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality];

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization];

    fn supported_devices(&self) -> &[DeviceClass];

    fn capabilities(&self) -> EngineCapabilities {
        let formats = self.supported_formats().to_vec();
        let modalities = formats
            .iter()
            .map(|format| (*format, self.supported_modalities(*format).to_vec()))
            .collect();
        let quantizations = formats
            .iter()
            .map(|format| (*format, self.supported_quantizations(*format).to_vec()))
            .collect();
        EngineCapabilities {
            formats,
            modalities,
            quantizations,
            devices: self.supported_devices().to_vec(),
        }
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>>;
}
