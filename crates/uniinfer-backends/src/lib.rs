//! Tokenization, text generation and realtime audio primitives.

pub mod audio;
pub mod chat_template;
pub mod output;
pub mod reasoning;
pub mod sampling;
pub mod stop;
pub mod text;
pub mod tokenizer;
pub mod tool_call;
pub mod utf8;

pub use chat_template::{ChatMessage, ChatTemplate, fallback_prompt};
pub use output::{AudioOutput, ImageOutput, ModelEvent, ModelOutput, VideoOutput};
pub use reasoning::{ReasoningEvent, ReasoningSplitter};
pub use sampling::Sampling;
pub use text::{FinishReason, GenerationOutput, GenerationSpec, TextGenerator};
pub use tokenizer::{ByteTokenizer, HuggingFaceTokenizer, Tokenizer};
pub use tool_call::{
    ParsedToolCalls, ToolCall, ToolStream, ToolStreamEvent, parse as parse_tool_calls,
};
