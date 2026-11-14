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

    /// Resolve a placement plan for one engine from the devices it supports.
    pub fn plan(
        &self,
        manifest: &ModelManifest,
        engine_devices: &[DeviceClass],
    ) -> Result<PlacementPlan> {
        let policy = manifest.placement.policy;
        let preferred = if manifest.placement.prefer.is_empty() {
            vec![DeviceClass::Npu, DeviceClass::Gpu, DeviceClass::Cpu]
        } else {
            manifest.placement.prefer.clone()
        };

        let mut available = Vec::new();
        for class in &preferred {
            if !engine_devices.contains(class) {
                continue;
            }
            if let Some(device) = self.find(*class) {
                available.push(device.clone());
            }
        }
        if available.is_empty() {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "engine supports {engine_devices:?}, available devices are {:?}",
                self.devices
                    .iter()
                    .map(|device| device.class)
                    .collect::<Vec<_>>()
            )));
        }

        if policy == PlacementPolicy::Strict
            && !available
                .iter()
                .any(|device| Some(device.class) == preferred.first().copied())
        {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "strict requires {}, which is unavailable",
                preferred[0]
            )));
        }

        let selected = &available[0];
        let fallback_reason = (selected.class != preferred[0]).then(|| {
            if !engine_devices.contains(&preferred[0]) {
                format!("engine_no_{}", preferred[0])
            } else {
                format!("device_unavailable_{}", preferred[0])
            }
        });
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
            if !engine_devices.contains(class) {
                return Err(EngineError::PlacementUnsatisfiable(format!(
                    "stage {stage} requests {class}, unsupported by the engine"
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
            fallback_reason,
        })
    }
}

impl From<Vec<Device>> for DeviceInventory {
    fn from(devices: Vec<Device>) -> Self {
        Self { devices }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use uniinfer_core::{Device, Modality, ModelFile, ModelFormat, PlacementConfig, Quantization};

    fn manifest(placement: PlacementConfig) -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            id: "test".into(),
            format: ModelFormat::Onnx,
            quantization: Quantization::Fp32,
            modalities: vec![Modality::Text],
            streaming_capable: false,
            context_len: 64,
            tokenizer: Some("bytes".into()),
            min_vram: None,
            placement,
            files: vec![ModelFile {
                path: PathBuf::from("model.onnx"),
                sha256: "0".repeat(64),
            }],
            text: None,
        }
    }

    fn config(policy: PlacementPolicy) -> PlacementConfig {
        PlacementConfig {
            policy,
            allow_cpu_fallback: true,
            prefer: Vec::new(),
            stages: BTreeMap::new(),
            offload_layers: None,
            kv_cache: None,
        }
    }

    fn gpu_inventory() -> DeviceInventory {
        DeviceInventory::from(vec![
            Device {
                id: "cpu:0".into(),
                class: DeviceClass::Cpu,
                backend: "cpu".into(),
                memory_bytes: 0,
            },
            Device {
                id: "gpu:0".into(),
                class: DeviceClass::Gpu,
                backend: "test".into(),
                memory_bytes: 0,
            },
        ])
    }

    #[test]
    fn auto_prefers_gpu_when_npu_is_absent() {
        let plan = gpu_inventory()
            .plan(
                &manifest(config(PlacementPolicy::Auto)),
                &[DeviceClass::Cpu, DeviceClass::Gpu],
            )
            .expect("plan");
        assert_eq!(plan.devices[0].class, DeviceClass::Gpu);
        assert_eq!(plan.stages["default"], DeviceClass::Gpu);
        assert_eq!(plan.fallback_reason.as_deref(), Some("engine_no_npu"));
    }

    #[test]
    fn auto_prefers_npu_over_gpu() {
        let inventory = DeviceInventory::from(vec![
            Device {
                id: "cpu:0".into(),
                class: DeviceClass::Cpu,
                backend: "cpu".into(),
                memory_bytes: 0,
            },
            Device {
                id: "gpu:0".into(),
                class: DeviceClass::Gpu,
                backend: "gpu".into(),
                memory_bytes: 0,
            },
            Device {
                id: "npu:0".into(),
                class: DeviceClass::Npu,
                backend: "npu".into(),
                memory_bytes: 0,
            },
        ]);
        let plan = inventory
            .plan(
                &manifest(config(PlacementPolicy::Auto)),
                &[DeviceClass::Cpu, DeviceClass::Gpu, DeviceClass::Npu],
            )
            .expect("plan");
        assert_eq!(plan.devices[0].class, DeviceClass::Npu);
        assert_eq!(plan.fallback_reason, None);
    }

    #[test]
    fn auto_falls_back_to_cpu_for_cpu_only_engine() {
        let plan = gpu_inventory()
            .plan(
                &manifest(config(PlacementPolicy::Auto)),
                &[DeviceClass::Cpu],
            )
            .expect("plan");
        assert_eq!(plan.devices[0].class, DeviceClass::Cpu);
    }

    #[test]
    fn strict_requires_preferred_device() {
        let error = gpu_inventory()
            .plan(
                &manifest(config(PlacementPolicy::Strict)),
                &[DeviceClass::Cpu],
            )
            .expect_err("must reject");
        assert!(error.to_string().contains("strict requires npu"));
    }

    #[test]
    fn pinned_requires_stages() {
        let error = gpu_inventory()
            .plan(
                &manifest(config(PlacementPolicy::Pinned)),
                &[DeviceClass::Cpu, DeviceClass::Gpu],
            )
            .expect_err("must reject");
        assert!(error.to_string().contains("placement.stages"));
    }

    #[test]
    fn stage_unsupported_by_engine_is_rejected() {
        let mut placement = config(PlacementPolicy::Pinned);
        placement.stages.insert("llm".into(), DeviceClass::Npu);
        let error = gpu_inventory()
            .plan(&manifest(placement), &[DeviceClass::Cpu, DeviceClass::Gpu])
            .expect_err("must reject");
        assert!(error.to_string().contains("unsupported by the engine"));
    }

    #[test]
    fn unsupported_offload_is_rejected() {
        let mut placement = config(PlacementPolicy::Auto);
        placement.offload_layers = Some("4".into());
        let error = gpu_inventory()
            .plan(&manifest(placement), &[DeviceClass::Cpu])
            .expect_err("must reject");
        assert!(error.to_string().contains("offload_layers=4"));
    }
}
