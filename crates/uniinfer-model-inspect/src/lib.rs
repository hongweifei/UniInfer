//! Model layout detection and manifest generation.
//!
//! This is intentionally engine-neutral: it reads files and metadata, then
//! emits a `ModelManifest` draft. GUI/CLI can show the draft before writing.

mod error;
mod files;
mod gguf;
mod inspect;
mod inspection;
mod metadata;
mod onnx;

pub use error::{InspectError, Result};
pub use inspect::{InspectOptions, inspect};
pub use inspection::Inspection;
