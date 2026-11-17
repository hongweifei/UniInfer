//! Host GPU inventory used for backend-variant compatibility hints.

use sysinfo::Gpus;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostGpu {
    pub vendor: String,
    pub model: String,
    pub memory_bytes: u64,
}

/// Enumerate physical GPUs independently of the inference execution providers
/// that happen to be compiled into this build.
#[must_use]
pub fn host_gpus() -> Vec<HostGpu> {
    let Ok(gpus) = Gpus::new_with_refreshed_list() else {
        return Vec::new();
    };
    gpus.iter()
        .map(|gpu| HostGpu {
            vendor: gpu.vendor().unwrap_or_default().to_owned(),
            model: gpu.model().unwrap_or_default().to_owned(),
            memory_bytes: gpu.total_memory().unwrap_or(0),
        })
        .collect()
}
