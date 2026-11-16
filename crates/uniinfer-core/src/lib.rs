//! Engine contracts, model manifests and shared types.

pub mod cancel;
pub mod device;
pub mod engine;
pub mod error;
pub mod manifest;
pub mod tensor;

pub use cancel::CancelToken;
pub use device::{
    Device, DeviceBudget, DeviceClass, KvCachePlacement, PlacementPlan, PlacementPolicy,
};
pub use engine::{
    EngineCapabilities, EngineLoadSpec, EngineSession, InferenceEngine, NativeFinish,
    NativeTextDecoder, NativeTextOutput, NativeTextRequest,
};
pub use error::{EngineError, Result};
pub use manifest::{
    AudioConfig, Capability, Modality, ModelFile, ModelFormat, ModelManifest, PlacementConfig,
    Quantization, Runtime, TextConfig,
};
pub use tensor::{Tensor, TensorData, TensorDtype, TensorSpec};
