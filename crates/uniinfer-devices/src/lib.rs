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
        let duplicate = self.devices.iter().any(|known| {
            known.id == device.id
                || (known.class == device.class && device.class == DeviceClass::Cpu)
        });
        if !duplicate {
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
        let policy = manifest.placement.policy;

        if policy == PlacementPolicy::Strict {
            let first = requested.first().ok_or_else(|| {
                EngineError::PlacementUnsatisfiable("strict requires supported_devices".into())
            })?;
            if self.find(*first).is_none() {
                return Err(EngineError::PlacementUnsatisfiable(format!(
                    "strict requires {first}, which is unavailable"
                )));
            }
        }

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
            if policy == PlacementPolicy::Pinned {
                return Err(EngineError::PlacementUnsatisfiable(
                    "pinned policy requires placement.stages".into(),
                ));
            }
            stages.insert("default".into(), selected.class);
        }

        for (stage, class) in &stages {
            if !requested.contains(class) {
                return Err(EngineError::PlacementUnsatisfiable(format!(
                    "stage {stage} requests {class}, outside supported_devices"
                )));
            }
            if self.find(*class).is_none() {
                return Err(EngineError::PlacementUnsatisfiable(format!(
                    "stage {stage} requests {class}, which is unavailable"
                )));
            }
        }

        let offload_layers = match manifest.placement.offload_layers.as_deref() {
            None | Some("0") => 0,
            Some(value) => {
                return Err(EngineError::PlacementUnsatisfiable(format!(
                    "offload_layers={value} is not implemented"
                )));
            }
        };

        let mut devices = vec![selected.clone()];
        for class in stages.values() {
            if let Some(device) = self.find(*class)
                && !devices.iter().any(|known| known.id == device.id)
            {
                devices.push(device.clone());
            }
        }

        Ok(PlacementPlan {
            policy,
            stages,
            devices,
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
