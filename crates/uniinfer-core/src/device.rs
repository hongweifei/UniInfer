use std::fmt;

use serde::{Deserialize, Serialize};

/// Device classes that participate in placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceClass {
    Cpu,
    Gpu,
    Npu,
}

impl fmt::Display for DeviceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Npu => "npu",
        };
        f.write_str(name)
    }
}

/// A concrete device discovered at startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Device {
    /// Stable identifier such as `cpu:0` or `gpu:cuda:0`.
    pub id: String,
    pub class: DeviceClass,
    /// Backend or execution provider name, e.g. `cpu`, `cuda`, `qnn`.
    pub backend: String,
    /// Usable memory in bytes, zero when unknown.
    pub memory_bytes: u64,
}

/// Resources available to a single engine load.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceBudget {
    pub devices: Vec<Device>,
}

impl DeviceBudget {
    #[must_use]
    pub fn has_device(&self, class: DeviceClass) -> bool {
        self.devices.iter().any(|device| device.class == class)
    }
}

/// Placement policy resolved from the model manifest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlacementPolicy {
    #[default]
    Auto,
    Pinned,
    Strict,
}

/// Where the KV cache lives for a session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCachePlacement {
    #[default]
    Vram,
    Ram,
    Auto,
}

/// Frozen placement decision for a loaded session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub policy: PlacementPolicy,
    /// Stage name to device class, e.g. `llm -> gpu`.
    pub stages: std::collections::BTreeMap<String, DeviceClass>,
    /// Devices selected for execution.
    pub devices: Vec<Device>,
    /// Transformer layers left on CPU, zero for fully resident models.
    pub offload_layers: usize,
    pub kv_cache: KvCachePlacement,
    /// Why a preference was not honored, e.g. `engine_no_npu`.
    pub fallback_reason: Option<String>,
}
