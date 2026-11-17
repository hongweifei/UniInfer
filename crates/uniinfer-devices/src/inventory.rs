//! Device discovery and registry.

use uniinfer_core::{Device, DeviceBudget, DeviceClass};

/// Devices discovered on the host.
#[derive(Debug, Clone, Default)]
pub struct DeviceInventory {
    pub(crate) devices: Vec<Device>,
}

impl DeviceInventory {
    /// Probe the host. The CPU is always available; GPU and NPU devices are registered
    /// by engines as execution providers become available.
    #[must_use]
    pub fn probe() -> Self {
        let parallelism =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let (total_memory, available_memory) =
            sysinfo::System::new().map_or((0, 0), |mut system| {
                system.refresh_memory();
                (system.total_memory(), system.available_memory())
            });
        Self {
            devices: vec![Device {
                id: "cpu:0".into(),
                class: DeviceClass::Cpu,
                backend: format!("cpu/{parallelism}t"),
                memory_bytes: total_memory,
                available_memory_bytes: available_memory,
                unified_memory: false,
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
        self.devices
            .iter()
            .filter(|device| device.class == class)
            // A dedicated accelerator beats an integrated one; within the same
            // memory architecture prefer the device with the most headroom.
            .max_by_key(|device| (!device.unified_memory, device.available_memory_bytes))
    }
}

impl From<Vec<Device>> for DeviceInventory {
    fn from(devices: Vec<Device>) -> Self {
        Self { devices }
    }
}
