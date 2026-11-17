//! Candle device probing and classification.

use candle_core::Device;
use uniinfer_core::{Device as CoreDevice, DeviceClass, EngineError, Result};

pub(crate) fn probe_devices() -> Vec<DeviceClass> {
    let mut devices = vec![DeviceClass::Cpu];
    if gpu_available() {
        devices.push(DeviceClass::Gpu);
    }
    devices
}

pub(crate) fn probe_device_ids(classes: &[DeviceClass]) -> Vec<CoreDevice> {
    classes
        .iter()
        .map(|class| CoreDevice {
            id: format!("{class}:candle:0"),
            class: *class,
            backend: "candle".into(),
            memory_bytes: 0,
            available_memory_bytes: 0,
            unified_memory: *class == DeviceClass::Gpu
                && cfg!(all(target_os = "macos", feature = "metal")),
        })
        .collect()
}

fn gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    if Device::new_cuda(0).is_ok() {
        return true;
    }
    #[cfg(feature = "metal")]
    if Device::new_metal(0).is_ok() {
        return true;
    }
    false
}

pub(crate) fn device_for(class: DeviceClass) -> Result<Device> {
    match class {
        DeviceClass::Cpu => Ok(Device::Cpu),
        DeviceClass::Gpu => {
            #[cfg(feature = "cuda")]
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }
            #[cfg(feature = "metal")]
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
            Err(EngineError::PlacementUnsatisfiable(
                "candle GPU backend is unavailable".into(),
            ))
        }
        DeviceClass::Npu => Err(EngineError::PlacementUnsatisfiable(
            "candle has no NPU backend".into(),
        )),
    }
}
