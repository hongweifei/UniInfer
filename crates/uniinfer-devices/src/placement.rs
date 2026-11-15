//! Placement policy resolution.

use uniinfer_core::{
    DeviceClass, EngineError, KvCachePlacement, ModelManifest, PlacementPlan, PlacementPolicy,
    Result,
};

use crate::DeviceInventory;

impl DeviceInventory {
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
