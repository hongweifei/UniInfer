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
use uniinfer_engine_onnx::OnnxEngine;
use uniinfer_registry::EngineRegistry;

/// Build the default engine registry and probe every engine's devices.
pub fn build() -> Result<EngineRegistry> {
    let onnx = OnnxEngine::new()?;
    let candle = CandleEngine::new()?;
    #[cfg(feature = "llama")]
    let llama = LlamaEngine::new()?;

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

    let mut registry = EngineRegistry::new(inventory);
    registry.register(Arc::new(onnx));
    #[cfg(feature = "llama")]
    registry.register(Arc::new(llama));
    registry.register(Arc::new(candle));
    Ok(registry)
}
