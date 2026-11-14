//! Device discovery and placement planning.

use uniinfer_core::{
    Device, DeviceBudget, DeviceClass, EngineError, KvCachePlacement, ModelManifest, PlacementPlan,
    PlacementPolicy, Result,
};

/// Devices discovered on the host.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    devices: Vec<Device>,
}

impl DeviceInventory {
    /// Probe the host. M0 reports the CPU; GPU and NPU devices are registered
    /// by engines as execution providers become available.
    #[must_use]
    pub fn probe() -> Self {
        let parallelism =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        Self {
            devices: vec![Device {
                id: "cpu:0".into(),
                class: DeviceClass::Cpu,
                backend: format!("cpu/{parallelism}t"),
                memory_bytes: 0,
            }],
        }
    }

    /// Register a device reported by an engine.
    pub fn register(&mut self, device: Device) {
        if !self.devices.iter().any(|known| known.id == device.id) {
            self.devices.push(device);
        }
    }

    #[must_use]
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    #[must_use]
    pub fn budget(&self) -> DeviceBudget {
        DeviceBudget {
            devices: self.devices.clone(),
        }
    }

    fn find(&self, class: DeviceClass) -> Option<&Device> {
        self.devices.iter().find(|device| device.class == class)
    }

    /// Resolve a placement plan from the manifest and available devices.
    pub fn plan(&self, manifest: &ModelManifest) -> Result<PlacementPlan> {
        let requested = if manifest.supported_devices.is_empty() {
            vec![DeviceClass::Cpu, DeviceClass::Gpu, DeviceClass::Npu]
        } else {
            manifest.supported_devices.clone()
        };

        let selected = requested
            .iter()
            .find_map(|class| self.find(*class))
            .ok_or_else(|| {
                EngineError::PlacementUnsatisfiable(format!(
                    "no available device among {requested:?}"
                ))
            })?;

        let mut stages = manifest.placement.stages.clone();
        if stages.is_empty() {
            stages.insert("default".into(), selected.class);
        }

        let offload_layers = manifest
            .placement
            .offload_layers
            .as_deref()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);

        Ok(PlacementPlan {
            policy: manifest.placement.policy,
            stages,
            devices: vec![selected.clone()],
            offload_layers,
            kv_cache: manifest
                .placement
                .kv_cache
                .unwrap_or(KvCachePlacement::Vram),
        })
    }
}

impl From<Vec<Device>> for DeviceInventory {
    fn from(devices: Vec<Device>) -> Self {
        Self { devices }
    }
}

/// Parse a placement policy name, used by tests and CLI flags.
pub fn parse_policy(value: &str) -> Result<PlacementPolicy> {
    match value {
        "auto" => Ok(PlacementPolicy::Auto),
        "pinned" => Ok(PlacementPolicy::Pinned),
        "strict" => Ok(PlacementPolicy::Strict),
        other => Err(EngineError::ManifestInvalid(format!(
            "unknown placement policy {other}"
        ))),
    }
}
