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
    /// Currently available memory in bytes, zero when unknown.
    #[serde(default)]
    pub available_memory_bytes: u64,
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
    /// Let the engine choose how many layers to offload from the live budget.
    #[serde(default)]
    pub auto_offload: bool,
    pub kv_cache: KvCachePlacement,
    /// Why a preference was not honored, e.g. `engine_no_npu`.
    pub fallback_reason: Option<String>,
}

/// Bytes accounted to each residency tier for a loaded session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Residency {
    pub vram_bytes: u64,
    pub host_bytes: u64,
    /// Backing bytes mapped or read from disk.
    pub disk_bytes: u64,
    /// Layers placed on the accelerator.
    pub offload_layers: usize,
    pub total_layers: Option<usize>,
    /// True when the engine could not report exact residency.
    #[serde(default)]
    pub estimated: bool,
}

impl Residency {
    #[must_use]
    pub fn total_bytes(self) -> u64 {
        self.vram_bytes
            .saturating_add(self.host_bytes)
            .saturating_add(self.disk_bytes)
    }
}

/// Aggregate ledger view exposed by `/metrics`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencySummary {
    pub sessions: usize,
    pub vram_bytes: u64,
    pub host_bytes: u64,
    pub disk_bytes: u64,
}

/// Parse `min_vram` values such as `6GiB`, `512MiB`, `8GB`, or a raw byte count.
#[must_use]
pub fn parse_memory_size(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let split = value
        .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(split);
    let number = number.parse::<f64>().ok()?;
    if !number.is_finite() || number < 0.0 {
        return None;
    }
    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        "kb" => 1000.0,
        "mb" => 1000.0 * 1000.0,
        "gb" => 1000.0 * 1000.0 * 1000.0,
        _ => return None,
    };
    Some((number * multiplier).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::parse_memory_size;

    #[test]
    fn parses_memory_units() {
        assert_eq!(parse_memory_size("6GiB"), Some(6 * 1024 * 1024 * 1024));
        assert_eq!(parse_memory_size("512MiB"), Some(512 * 1024 * 1024));
        assert_eq!(parse_memory_size("8GB"), Some(8_000_000_000));
        assert_eq!(parse_memory_size("1024"), Some(1024));
        assert_eq!(parse_memory_size("manyGiB"), None);
        assert_eq!(parse_memory_size("-1GiB"), None);
    }
}
