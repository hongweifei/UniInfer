//! Chat message rendering from tokenizer templates.

use std::path::Path;

use minijinja::value::Serde;
use minijinja::{Environment, Value, context};
use serde::Serialize;
use uniinfer_core::{EngineError, Result};

/// Chat message passed to a tokenizer chat template.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Compiled chat template loaded from `tokenizer_config.json`.
pub struct ChatTemplate {
    environment: Environment<'static>,
    supports_thinking: bool,
}

impl ChatTemplate {
    /// Load the optional `chat_template` field from a tokenizer config.
    pub fn from_tokenizer_config(path: &Path) -> Result<Option<Self>> {
        if !path.is_file() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path).map_err(|error| {
            EngineError::ManifestInvalid(format!("cannot read {}: {error}", path.display()))
        })?;
        let config: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
            EngineError::ManifestInvalid(format!("{}: {error}", path.display()))
        })?;
        let Some(template) = config
            .get("chat_template")
            .and_then(serde_json::Value::as_str)
        else {
            return Ok(None);
        };
        let supports_thinking = template.to_ascii_lowercase().contains("thinking")
            || template.to_ascii_lowercase().contains("think>");
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment
            .add_template_owned("chat", template.to_owned())
            .map_err(|error| {
                EngineError::ManifestInvalid(format!(
                    "chat template in {}: {error}",
                    path.display()
                ))
            })?;
        Ok(Some(Self {
            environment,
            supports_thinking,
        }))
    }

    /// Render messages with `add_generation_prompt = true`.
    pub fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        self.render_with_tools(messages, &[], None)
    }

    /// Render messages with optional OpenAI-style tool definitions.
    pub fn render_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        tool_choice: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.render_with_options(messages, tools, tool_choice, None)
    }

    /// Render with an optional `enable_thinking` template flag.
    pub fn render_with_options(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        tool_choice: Option<&serde_json::Value>,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let template = self
            .environment
            .get_template("chat")
            .map_err(|error| EngineError::Other(format!("chat template: {error}")))?;
        let thinking = enable_thinking.map(Value::from).unwrap_or(Value::UNDEFINED);
        template
            .render(context! {
                messages => Value::from(Serde(messages)),
                add_generation_prompt => true,
                tools => Value::from(Serde(tools)),
                tool_choice => tool_choice
                    .map(|value| Value::from(Serde(value)))
                    .unwrap_or(Value::from(())),
                enable_thinking => thinking.clone(),
                thinking => thinking,
            })
            .map_err(|error| EngineError::Other(format!("render chat template: {error}")))
    }

    #[must_use]
    pub fn supports_thinking(&self) -> bool {
        self.supports_thinking
    }
}

/// Fallback used by models without a tokenizer chat template.
pub fn fallback_prompt(messages: &[ChatMessage]) -> Result<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.clone())
        .filter(|content| !content.is_empty())
        .ok_or_else(|| {
            EngineError::ManifestInvalid(
                "chat messages must contain a non-empty user message".into(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_simple_template() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer_config.json");
        std::fs::write(
            &path,
            r#"{"chat_template":"{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"}"#,
        )
        .expect("write");
        let template = ChatTemplate::from_tokenizer_config(&path)
            .expect("load")
            .expect("template");
        let prompt = template
            .render(&[ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }])
            .expect("render");
        assert_eq!(prompt, "<user>hi<assistant>");
    }

    #[test]
    fn fallback_uses_last_user_message() {
        let prompt = fallback_prompt(&[
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: "answer".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ])
        .expect("prompt");
        assert_eq!(prompt, "second");
    }
    #[test]
    fn passes_enable_thinking_to_template() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer_config.json");
        std::fs::write(
            &path,
            r#"{"chat_template":"{% if enable_thinking is defined and enable_thinking %}think{% else %}plain{% endif %}"}"#,
        )
        .expect("config");
        let template = ChatTemplate::from_tokenizer_config(&path)
            .expect("load")
            .expect("template");
        assert!(template.supports_thinking());
        assert_eq!(
            template
                .render_with_options(&[], &[], None, None)
                .expect("none"),
            "plain"
        );
        assert_eq!(
            template
                .render_with_options(&[], &[], None, Some(true))
                .expect("enabled"),
            "think"
        );
        assert_eq!(
            template
                .render_with_options(&[], &[], None, Some(false))
                .expect("disabled"),
            "plain"
        );
    }
}
