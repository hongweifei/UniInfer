//! Default engine composition shared by the CLI, API and desktop shell.
//!
//! Keeping engine construction here means `uniinfer-service` no longer
//! depends on concrete engines, and engine features are declared once.

use std::sync::Arc;

use uniinfer_core::Result;
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_candle::CandleEngine;
#[cfg(feature = "llama")]
use uniinfer_engine_llama::LlamaEngine;
#[cfg(feature = "llama-mtmd")]
use uniinfer_engine_llama::LlamaMtmdEngine;
use uniinfer_engine_onnx::OnnxEngine;
use uniinfer_engine_process::ProcessEngine;
#[cfg(feature = "whisper")]
use uniinfer_engine_whisper::WhisperEngine;
use uniinfer_registry::EngineRegistry;

/// Build the default engine registry and probe every engine's devices.
pub fn build() -> Result<EngineRegistry> {
    build_with_process(false)
}

/// Build the engine registry, optionally routing text models through an
/// isolated worker process.
pub fn build_with_process(isolate_models: bool) -> Result<EngineRegistry> {
    let onnx = OnnxEngine::new()?;
    let candle = CandleEngine::new()?;
    #[cfg(feature = "llama")]
    let llama = LlamaEngine::new()?;
    #[cfg(feature = "llama-mtmd")]
    let mtmd = LlamaMtmdEngine::new()?;
    #[cfg(feature = "whisper")]
    let whisper = WhisperEngine::new();

    let mut inventory = DeviceInventory::probe();
    for device in onnx.probe_devices() {
        inventory.register(device);
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

    let mut registry = EngineRegistry::new(inventory);
    if isolate_models {
        registry.register(Arc::new(ProcessEngine::new()));
    }
    registry.register(Arc::new(onnx));
    #[cfg(feature = "llama-mtmd")]
    registry.register(Arc::new(mtmd));
    #[cfg(feature = "llama")]
    registry.register(Arc::new(llama));
    #[cfg(feature = "whisper")]
    registry.register(Arc::new(whisper));
    registry.register(Arc::new(candle));
    Ok(registry)
}
