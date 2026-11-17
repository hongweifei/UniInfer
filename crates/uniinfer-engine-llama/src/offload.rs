//! Choose llama.cpp model/context parameters from the placement budget.

use std::path::Path;

use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::model::params::LlamaModelParams;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, KvCachePlacement, Residency, Result,
};

const AUTO_RESERVE: u64 = 256 * 1024 * 1024;

pub(super) struct LlamaOffload {
    pub(super) params: LlamaModelParams,
    pub(super) offload_kqv: bool,
    pub(super) residency: Residency,
}

struct ModelMeta {
    layers: u64,
    bytes: u64,
    n_embd: u64,
    n_head: u64,
    n_head_kv: u64,
}

pub(super) fn plan(spec: &EngineLoadSpec, path: &Path) -> Result<LlamaOffload> {
    let class = spec
        .placement
        .devices
        .first()
        .map(|device| device.class)
        .unwrap_or(DeviceClass::Cpu);
    let (meta, estimated) = match read_metadata(path) {
        Some(meta) if meta.layers > 0 => (meta, false),
        Some(meta) if !spec.placement.auto_offload => (meta, true),
        Some(_) | None if spec.placement.auto_offload => {
            return Err(EngineError::LoadFailed(format!(
                "GGUF metadata missing layer count: {}",
                path.display()
            )));
        }
        _ => {
            let fallback = ModelMeta {
                layers: u64::try_from(spec.placement.offload_layers.max(1)).unwrap_or(1),
                bytes: std::fs::metadata(path).map_or(0, |metadata| metadata.len()),
                n_embd: 0,
                n_head: 1,
                n_head_kv: 0,
            };
            (fallback, true)
        }
    };
    let kv_on_gpu = class == DeviceClass::Gpu && spec.placement.kv_cache != KvCachePlacement::Ram;
    let n_gpu_layers = if class == DeviceClass::Gpu {
        let requested = if spec.placement.auto_offload {
            auto_layers(spec, &meta, kv_on_gpu)?
        } else {
            u32::try_from(spec.placement.offload_layers).unwrap_or(u32::MAX)
        };
        requested.min(u32::try_from(meta.layers).unwrap_or(u32::MAX))
    } else {
        0
    };
    let offload_kqv = kv_on_gpu && n_gpu_layers > 0;
    let kv_bytes = estimate_kv(spec.manifest.context_len, &meta);
    let weights_on_gpu = meta.bytes.saturating_mul(u64::from(n_gpu_layers)) / meta.layers.max(1);
    let residency = Residency {
        vram_bytes: weights_on_gpu + if offload_kqv { kv_bytes } else { 0 },
        host_bytes: if offload_kqv { 0 } else { kv_bytes },
        disk_bytes: meta.bytes,
        offload_layers: usize::try_from(n_gpu_layers).unwrap_or(usize::MAX),
        total_layers: Some(usize::try_from(meta.layers).unwrap_or(usize::MAX)),
        estimated,
    };
    let mut params = LlamaModelParams::default().with_use_mmap(true);
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
        vram = residency.vram_bytes,
        host = residency.host_bytes,
        disk = residency.disk_bytes,
        estimated,
        "llama.cpp offload plan"
    );
    Ok(LlamaOffload {
        params,
        offload_kqv,
        residency,
    })
}

fn read_metadata(path: &Path) -> Option<ModelMeta> {
    let context = GgufContext::from_file(path)?;
    let architecture = get_string(&context, "general.architecture").unwrap_or("llama");
    let layers = get_u64(&context, &format!("{architecture}.block_count"))?;
    let n_embd = get_u64(&context, &format!("{architecture}.embedding_length")).unwrap_or(0);
    let n_head = get_u64(&context, &format!("{architecture}.attention.head_count"))
        .unwrap_or(1)
        .max(1);
    let n_head_kv =
        get_u64(&context, &format!("{architecture}.attention.head_count_kv")).unwrap_or(n_head);
    Some(ModelMeta {
        layers,
        bytes: std::fs::metadata(path).map_or(0, |metadata| metadata.len()),
        n_embd,
        n_head,
        n_head_kv,
    })
}

fn get_u64(context: &GgufContext, key: &str) -> Option<u64> {
    let index = context.find_key(key);
    if index < 0 {
        return None;
    }
    Some(u64::from(context.val_u32(index)))
}

fn get_string<'a>(context: &'a GgufContext, key: &str) -> Option<&'a str> {
    let index = context.find_key(key);
    if index < 0 {
        return None;
    }
    context.val_str(index)
}

fn auto_layers(spec: &EngineLoadSpec, meta: &ModelMeta, offload_kqv: bool) -> Result<u32> {
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
        estimate_kv(spec.manifest.context_len, meta)
    } else {
        0
    };
    Ok(layers_for_budget(
        meta.bytes,
        meta.layers,
        available,
        kv_bytes,
        AUTO_RESERVE,
    ))
}

fn estimate_kv(context_len: usize, meta: &ModelMeta) -> u64 {
    let context = u64::try_from(context_len.max(1)).unwrap_or(u64::MAX);
    let head_dim = meta.n_embd.checked_div(meta.n_head).unwrap_or(0);
    // K and V, f16 elements, for the configured context.
    4u64.saturating_mul(meta.layers)
        .saturating_mul(context)
        .saturating_mul(meta.n_head_kv.max(1))
        .saturating_mul(head_dim)
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
