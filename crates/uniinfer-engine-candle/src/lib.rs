//! Candle engine with manifest-driven text family registry.
//!
//! `CandleEngine` owns routing and device selection; concrete model families
//! live under `text/` and register themselves. Session state, device probing
//! and cancellation IO are separate modules.

mod cancel_io;
mod device;
mod engine;
mod session;
mod text;

pub use engine::CandleEngine;
