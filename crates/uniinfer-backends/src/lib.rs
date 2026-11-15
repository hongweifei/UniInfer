//! Tokenization, text generation and realtime audio primitives.

pub mod audio;
pub mod chat_template;
pub mod stop;
pub mod text;
pub mod tokenizer;
pub mod utf8;

pub use chat_template::{ChatMessage, ChatTemplate, fallback_prompt};
pub use text::{FinishReason, GenerationOutput, TextGenerator};
pub use tokenizer::{ByteTokenizer, HuggingFaceTokenizer, Tokenizer};
