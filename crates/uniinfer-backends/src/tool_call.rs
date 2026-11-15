//! Tool-call parsing from generated text.
//!
//! Formats stay outside generation: the model emits text, this module
//! recognizes the common Qwen3/Hermes blocks and a plain JSON object.

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCalls {
    /// Text outside tool-call blocks; `None` when only a call was emitted.
    pub content: Option<String>,
    pub calls: Vec<ToolCall>,
}

#[must_use]
pub fn parse(content: &str, id_prefix: &str) -> ParsedToolCalls {
    let mut calls = Vec::new();
    let mut outside = String::new();
    let mut rest = content;

    while let Some(start) = rest.find("<tool_call>") {
        outside.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            outside.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let raw = after[..end].trim();
        if let Some(call) = parse_call(raw, id_prefix, calls.len()) {
            calls.push(call);
        } else {
            outside
                .push_str(&rest[start..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    outside.push_str(rest);

    if calls.is_empty()
        && let Some(marker) = outside.find("[TOOL_CALLS]")
    {
        let payload = outside[marker + "[TOOL_CALLS]".len()..].trim();
        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            let values = match value {
                Value::Array(values) => values,
                value => vec![value],
            };
            for (index, value) in values.into_iter().enumerate() {
                if let Some(call) = parse_call_value(&value, id_prefix, index) {
                    calls.push(call);
                }
            }
            if !calls.is_empty() {
                outside.truncate(marker);
            }
        }
    }

    if calls.is_empty() {
        if let Some(call) = parse_call(content.trim(), id_prefix, 0) {
            calls.push(call);
            outside.clear();
        }
    }

    let content = outside.trim();
    ParsedToolCalls {
        content: (!content.is_empty()).then(|| content.to_owned()),
        calls,
    }
}

fn parse_call(raw: &str, prefix: &str, index: usize) -> Option<ToolCall> {
    let value: Value = serde_json::from_str(raw).ok()?;
    parse_call_value(&value, prefix, index)
}

fn parse_call_value(value: &Value, prefix: &str, index: usize) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?.to_owned();
    let arguments = match value.get("arguments")? {
        Value::String(arguments) => arguments.clone(),
        arguments => arguments.to_string(),
    };
    Some(ToolCall {
        id: format!("call_{prefix}_{index}"),
        kind: "function",
        function: FunctionCall { name, arguments },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_blocks() {
        let parsed = parse(
            "thinking\n<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n</tool_call>",
            "s1",
        );
        assert_eq!(parsed.content.as_deref(), Some("thinking"));
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].function.name, "lookup");
        assert_eq!(parsed.calls[0].function.arguments, "{\"q\":\"x\"}");
    }

    #[test]
    fn parses_plain_json_call() {
        let parsed = parse(r#"{"name":"ping","arguments":"{}"}"#, "s2");
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.calls[0].function.name, "ping");
    }

    #[test]
    fn parses_mistral_array() {
        let parsed = parse(
            "calling\n[TOOL_CALLS] [{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}]",
            "s4",
        );
        assert_eq!(parsed.content.as_deref(), Some("calling"));
        assert_eq!(parsed.calls[0].function.name, "lookup");
    }

    #[test]
    fn leaves_plain_text_untouched() {
        let parsed = parse("hello", "s3");
        assert_eq!(parsed.calls.len(), 0);
        assert_eq!(parsed.content.as_deref(), Some("hello"));
    }
}
