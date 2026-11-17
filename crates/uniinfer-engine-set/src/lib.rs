//! Default engine composition shared by the CLI, API and desktop shell.
//!
//! Keeping engine construction here means `uniinfer-service` no longer
//! depends on concrete engines, and engine features are declared once.

use std::sync::Arc;

use uniinfer_backend_store::{BackendLauncher, BackendRuntime};
use uniinfer_core::Result;
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_candle::CandleEngine;
#[cfg(feature = "llama")]
use uniinfer_engine_llama::LlamaEngine;
#[cfg(feature = "llama-mtmd")]
use uniinfer_engine_llama::LlamaMtmdEngine;
use uniinfer_engine_llama_server::LlamaServerEngine;
use uniinfer_engine_onnx::OnnxEngine;
use uniinfer_engine_process::ProcessEngine;
use uniinfer_engine_sd_server::StableDiffusionServerEngine;
#[cfg(feature = "whisper")]
use uniinfer_engine_whisper::WhisperEngine;
use uniinfer_engine_whisper_server::WhisperServerEngine;
use uniinfer_registry::EngineRegistry;

/// Build the default engine registry and probe every engine's devices.
pub fn build() -> Result<EngineRegistry> {
    build_with_process(false)
}

/// Build the engine registry, optionally routing text models through an
/// isolated worker process.
pub fn build_with_process(isolate_models: bool) -> Result<EngineRegistry> {
    build_with_backend(isolate_models, BackendRuntime::Builtin)
}

/// Build the engine registry with an explicit backend composition. Built-in
/// engines are always registered; sidecars are selected per model manifest.
pub fn build_with_backend(isolate_models: bool, backend: BackendRuntime) -> Result<EngineRegistry> {
    let sidecars = match backend {
        BackendRuntime::Builtin => Vec::new(),
        BackendRuntime::Sidecar(backend) => vec![backend],
        BackendRuntime::Sidecars(backends) => backends,
    };
    build_with_backends(isolate_models, &sidecars)
}

/// Build the registry with every installed sidecar variant. Built-in engines
/// are registered first; a model manifest selects a sidecar explicitly.
pub fn build_with_backends(
    isolate_models: bool,
    sidecars: &[uniinfer_backend_store::BackendLaunch],
) -> Result<EngineRegistry> {
    #[cfg(feature = "onnx-dynamic")]
    let onnx_backend = sidecars
        .iter()
        .find(|backend| backend.launcher == BackendLauncher::OnnxRuntime);
    #[cfg(feature = "onnx-dynamic")]
    if let Some(backend) = onnx_backend {
        uniinfer_engine_onnx::configure_dynamic_library(&backend.worker)?;
    }
    #[cfg(feature = "onnx-dynamic")]
    let onnx = match onnx_backend {
        Some(backend) => OnnxEngine::with_identity(&backend.family, &backend.variant),
        None => OnnxEngine::new(),
    };
    #[cfg(not(feature = "onnx-dynamic"))]
    let onnx = OnnxEngine::new();
    let onnx = match onnx {
        Ok(engine) => Some(engine),
        Err(error) if cfg!(feature = "onnx-dynamic") => {
            tracing::warn!(%error, "ONNX Runtime unavailable; install an ONNX Runtime backend");
            None
        }
        Err(error) => return Err(error),
    };
    let candle = CandleEngine::new()?;
    #[cfg(feature = "llama")]
    let llama = LlamaEngine::new()?;
    #[cfg(feature = "llama-mtmd")]
    let mtmd = LlamaMtmdEngine::new()?;
    #[cfg(feature = "whisper")]
    let whisper = WhisperEngine::new();

    let mut inventory = DeviceInventory::probe();
    if let Some(onnx) = &onnx {
        for device in onnx.probe_devices() {
            inventory.register(device);
        }
    }
    for device in candle.probe_devices() {
        inventory.register(device);
    }
    #[cfg(feature = "llama")]
    for device in llama.probe_devices() {
        inventory.register(device);
    }
    #[cfg(feature = "llama-mtmd")]
    for device in mtmd.probe_devices() {
        inventory.register(device);
    }
    #[cfg(feature = "whisper")]
    for device in whisper.probe_devices() {
        inventory.register(device);
    }

    let registry = EngineRegistry::new(inventory);
    if isolate_models {
        registry.register(Arc::new(ProcessEngine::new()));
    }
    if let Some(onnx) = onnx {
        registry.register(Arc::new(onnx));
    }
    #[cfg(feature = "llama-mtmd")]
    registry.register(Arc::new(mtmd));
    #[cfg(feature = "llama")]
    registry.register(Arc::new(llama));
    #[cfg(feature = "whisper")]
    registry.register(Arc::new(whisper));
    registry.register(Arc::new(candle));

    register_sidecars(&registry, sidecars);
    Ok(registry)
}

/// Register sidecars that appeared after the runtime was built. An existing
/// family/variant pair is left untouched.
pub fn register_sidecars(
    registry: &EngineRegistry,
    sidecars: &[uniinfer_backend_store::BackendLaunch],
) {
    let mut seen = std::collections::HashSet::new();
    for backend in sidecars {
        let key = (backend.family.clone(), backend.variant.to_ascii_lowercase());
        if !seen.insert(key) {
            continue;
        }
        register_sidecar(registry, backend);
    }
}

fn register_sidecar(registry: &EngineRegistry, backend: &uniinfer_backend_store::BackendLaunch) {
    match backend.launcher {
        BackendLauncher::Worker => {
            registry.register_sidecar(Arc::new(ProcessEngine::with_identity(
                backend.worker.clone(),
                backend.devices.clone(),
                backend.family.clone(),
                backend.variant.clone(),
            )));
        }
        BackendLauncher::LlamaServer => {
            registry.register_sidecar(Arc::new(LlamaServerEngine::with_identity(
                backend.worker.clone(),
                backend.devices.clone(),
                backend.family.clone(),
                backend.variant.clone(),
            )));
        }
        BackendLauncher::StableDiffusionServer => {
            registry.register_sidecar(Arc::new(StableDiffusionServerEngine::with_identity(
                backend.worker.clone(),
                backend.devices.clone(),
                backend.family.clone(),
                backend.variant.clone(),
            )));
        }
        BackendLauncher::WhisperServer => {
            registry.register_sidecar(Arc::new(WhisperServerEngine::with_identity(
                backend.worker.clone(),
                backend.devices.clone(),
                backend.family.clone(),
                backend.variant.clone(),
            )));
        }
        BackendLauncher::OnnxRuntime => {
            #[cfg(not(feature = "onnx-dynamic"))]
            tracing::warn!(
                id = %backend.id,
                "ONNX Runtime package requires an onnx-dynamic build"
            );
        }
    }
}
