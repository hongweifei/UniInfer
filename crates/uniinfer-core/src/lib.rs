//! Engine contracts, model manifests and shared types.

pub mod device;
pub mod engine;
pub mod error;
pub mod manifest;
pub mod tensor;

pub use device::{
    Device, DeviceBudget, DeviceClass, KvCachePlacement, PlacementPlan, PlacementPolicy,
};
pub use engine::{EngineCapabilities, EngineLoadSpec, EngineSession, InferenceEngine};
pub use error::{EngineError, Result};
pub use manifest::{
    Modality, ModelFile, ModelFormat, ModelManifest, PlacementConfig, Quantization, TextConfig,
};
pub use tensor::{Tensor, TensorData, TensorDtype, TensorSpec};
