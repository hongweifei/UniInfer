use uniinfer_core::{Device, DeviceClass};

/// Rank a derived backend variant against the live device inventory. The
/// name comparison is token-based so catalog additions (SYCL, OpenVINO, NPU,
/// vendor-specific names) do not need a matching allowlist here.
pub(super) fn variant_rank(devices: &[Device], variant: &str) -> u8 {
    let variant = variant.to_ascii_lowercase();
    let variant_tokens = tokens(&variant);
    if variant_tokens.contains(&"cpu") {
        return if devices
            .iter()
            .any(|device| device.class == DeviceClass::Cpu)
        {
            0
        } else {
            2
        };
    }
    let accelerators = devices
        .iter()
        .filter(|device| device.class != DeviceClass::Cpu)
        .collect::<Vec<_>>();
    if accelerators.is_empty() {
        return 2;
    }
    let matched = accelerators.iter().any(|device| {
        let backend = device.backend.to_ascii_lowercase();
        let backend_tokens = tokens(&backend);
        variant_tokens.iter().any(|variant_token| {
            backend_tokens.iter().any(|backend_token| {
                backend_token.contains(variant_token) || variant_token.contains(backend_token)
            })
        })
    });
    if matched { 0 } else { 3 }
}

fn tokens(value: &str) -> Vec<&str> {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(class: DeviceClass, backend: &str) -> Device {
        Device {
            id: format!("{class:?}:{backend}"),
            class,
            backend: backend.into(),
            memory_bytes: 0,
            available_memory_bytes: 0,
            unified_memory: false,
        }
    }

    #[test]
    fn ranks_matching_accelerators_above_fallbacks() {
        let devices = vec![
            device(DeviceClass::Cpu, "cpu"),
            device(DeviceClass::Gpu, "sycl"),
        ];
        assert_eq!(variant_rank(&devices, "SYCL"), 0);
        assert_eq!(variant_rank(&devices, "CUDA 12.4"), 3);
        assert_eq!(variant_rank(&devices, "CPU"), 0);
        assert_eq!(variant_rank(&devices[..1], "SYCL"), 2);
    }
}
