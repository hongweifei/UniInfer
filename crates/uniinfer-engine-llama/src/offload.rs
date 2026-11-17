//! Choose llama.cpp model/context parameters from the placement budget.

use std::path::Path;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use uniinfer_core::{DeviceClass, EngineError, EngineLoadSpec, KvCachePlacement, Result};

const AUTO_RESERVE: u64 = 256 * 1024 * 1024;

pub(super) struct LlamaOffload {
    pub(super) params: LlamaModelParams,
    pub(super) offload_kqv: bool,
}

pub(super) fn plan(
    spec: &EngineLoadSpec,
    path: &Path,
    backend: &LlamaBackend,
) -> Result<LlamaOffload> {
    let class = spec
        .placement
        .devices
        .first()
        .map(|device| device.class)
        .unwrap_or(DeviceClass::Cpu);
    let kv_on_gpu = class == DeviceClass::Gpu && spec.placement.kv_cache != KvCachePlacement::Ram;
    let mut params = LlamaModelParams::default().with_use_mmap(true);
    let n_gpu_layers = if class == DeviceClass::Gpu {
        if spec.placement.auto_offload {
            auto_layers(spec, path, backend, kv_on_gpu)?
        } else {
            u32::try_from(spec.placement.offload_layers).unwrap_or(u32::MAX)
        }
    } else {
        0
    };
    let offload_kqv = kv_on_gpu && n_gpu_layers > 0;
    if n_gpu_layers > 0 {
        params = params.with_n_gpu_layers(n_gpu_layers);
    }
    tracing::info!(
        model = %spec.manifest.id,
        class = %class,
        available = spec
            .placement
            .devices
            .first()
            .map_or(0, |device| device.available_memory_bytes),
        n_gpu_layers,
        offload_kqv,
        "llama.cpp offload plan"
    );
    Ok(LlamaOffload {
        params,
        offload_kqv,
    })
}

fn auto_layers(
    spec: &EngineLoadSpec,
    path: &Path,
    backend: &LlamaBackend,
    offload_kqv: bool,
) -> Result<u32> {
    let probe_params = LlamaModelParams::default()
        .with_vocab_only(true)
        .with_use_mmap(true);
    let probe = LlamaModel::load_from_file(backend, path, &probe_params).map_err(|error| {
        if spec.cancel.is_cancelled() {
            EngineError::Cancelled
        } else {
            EngineError::LoadFailed(format!("llama metadata {}: {error}", path.display()))
        }
    })?;
    let layers = u64::from(probe.n_layer());
    if layers == 0 {
        return Ok(0);
    }
    let model_bytes = match probe.size() {
        0 => std::fs::metadata(path).map_or(0, |metadata| metadata.len()),
        bytes => bytes,
    };
    let n_embd = u64::try_from(probe.n_embd()).unwrap_or(0);
    let n_head = u64::from(probe.n_head()).max(1);
    let n_head_kv = u64::from(probe.n_head_kv());
    let context = u64::try_from(spec.manifest.context_len.max(1)).unwrap_or(u64::MAX);
    drop(probe);
    if model_bytes == 0 {
        return Ok(0);
    }

    let Some(gpu) = spec
        .placement
        .devices
        .iter()
        .find(|device| device.class == DeviceClass::Gpu)
    else {
        return Ok(0);
    };
    let live_free = llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|device| {
            matches!(
                device.device_type,
                llama_cpp_2::LlamaBackendDeviceType::Gpu
                    | llama_cpp_2::LlamaBackendDeviceType::IntegratedGpu
                    | llama_cpp_2::LlamaBackendDeviceType::Accelerator
            )
        })
        .map(|device| u64::try_from(device.memory_free).unwrap_or(0))
        .max()
        .unwrap_or(0);
    let available = if live_free > 0 {
        live_free
    } else {
        gpu.available_memory_bytes
    };
    if available == 0 {
        tracing::warn!(
            model = %spec.manifest.id,
            "GPU free memory is unknown; auto offload stays on CPU"
        );
        return Ok(0);
    }

    let kv_bytes = if offload_kqv {
        let head_dim = n_embd / n_head;
        // K and V, f16 elements, for the configured context.
        4u64.saturating_mul(layers)
            .saturating_mul(context)
            .saturating_mul(n_head_kv.max(1))
            .saturating_mul(head_dim)
    } else {
        0
    };
    Ok(layers_for_budget(
        model_bytes,
        layers,
        available,
        kv_bytes,
        AUTO_RESERVE,
    ))
}

fn layers_for_budget(
    model_bytes: u64,
    layers: u64,
    available: u64,
    kv_bytes: u64,
    reserve: u64,
) -> u32 {
    if model_bytes == 0 || layers == 0 {
        return 0;
    }
    let usable = available.saturating_sub(reserve).saturating_sub(kv_bytes);
    let per_layer = model_bytes.div_ceil(layers);
    let n = (usable / per_layer).min(layers);
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::layers_for_budget;

    const MIB: u64 = 1024 * 1024;

    #[test]
    fn budgets_entire_model_when_vram_fits() {
        assert_eq!(
            layers_for_budget(4_000 * MIB, 32, 8_000 * MIB, 0, 256 * MIB),
            32
        );
    }

    #[test]
    fn offloads_a_prefix_for_a_larger_model() {
        assert_eq!(
            layers_for_budget(8_000 * MIB, 32, 4_000 * MIB, 0, 256 * MIB),
            14
        );
    }

    #[test]
    fn reserves_kv_space_before_choosing_layers() {
        assert_eq!(
            layers_for_budget(8_000 * MIB, 32, 4_000 * MIB, 512 * MIB, 256 * MIB),
            12
        );
    }

    #[test]
    fn keeps_everything_on_cpu_when_nothing_fits() {
        assert_eq!(
            layers_for_budget(8_000 * MIB, 32, 128 * MIB, 0, 256 * MIB),
            0
        );
    }
}
