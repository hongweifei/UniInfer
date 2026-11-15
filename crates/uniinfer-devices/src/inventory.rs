//! Device discovery and registry.

use uniinfer_core::{Device, DeviceBudget, DeviceClass};

/// Devices discovered on the host.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    pub(crate) devices: Vec<Device>,
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

    pub(crate) fn find(&self, class: DeviceClass) -> Option<&Device> {
        self.devices.iter().find(|device| device.class == class)
    }
}

impl From<Vec<Device>> for DeviceInventory {
    fn from(devices: Vec<Device>) -> Self {
        Self { devices }
    }
}
