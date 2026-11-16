//! Device discovery and placement planning.

mod inventory;
mod placement;

pub use inventory::DeviceInventory;

#[cfg(test)]
mod tests {

    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use uniinfer_core::{
        Capability, Device, DeviceClass, Modality, Mode, ModelFile, ModelFormat, ModelManifest,
        Operation, PlacementConfig, PlacementPolicy, Quantization, Runtime,
    };

    fn manifest(placement: PlacementConfig) -> ModelManifest {
        ModelManifest {
            id: "test".into(),
            runtime: Runtime::Onnx,
            implementation: "onnx.graph".into(),
            capabilities: vec![Capability {
                name: "text_chat".into(),
                operation: Operation::Generate,
                required_inputs: vec![Modality::Text],
                optional_inputs: Vec::new(),
                outputs: vec![Modality::Text],
                mode: Mode::Stream,
            }],
            format: ModelFormat::Onnx,
            quantization: Quantization::Fp32,
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
            audio: None,
            vision: None,
            pipeline: None,
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
