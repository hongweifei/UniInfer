//! Chat content parts and image URL decoding.

use base64::Engine as _;
use serde::Deserialize;

use crate::error::ApiError;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum ChatContent {
    Text(String),
    Parts(Vec<serde_json::Value>),
}

#[derive(Debug, Default)]
pub(super) struct ParsedContent {
    pub(super) text: String,
    pub(super) images: Vec<Vec<u8>>,
}

impl ChatContent {
    pub(super) fn contains_image(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(|part| {
                part.get("type").and_then(serde_json::Value::as_str) == Some("image_url")
            }),
        }
    }

    pub(super) fn into_parts(self, image_marker: &str) -> Result<ParsedContent, ApiError> {
        match self {
            Self::Text(text) => Ok(ParsedContent {
                text,
                images: Vec::new(),
            }),
            Self::Parts(parts) => {
                let mut content = ParsedContent::default();
                for part in parts {
                    match part.get("type").and_then(serde_json::Value::as_str) {
                        Some("text") => {
                            if let Some(value) =
                                part.get("text").and_then(serde_json::Value::as_str)
                            {
                                content.text.push_str(value);
                            }
                        }
                        Some("image_url") => {
                            if content.images.len() == uniinfer_core::MAX_IMAGES {
                                return Err(ApiError::bad_request(
                                    "too_many_images",
                                    format!(
                                        "at most {} images are supported per request",
                                        uniinfer_core::MAX_IMAGES
                                    ),
                                ));
                            }
                            content.images.push(decode_image_part(&part)?);
                            content.text.push_str(image_marker);
                        }
                        Some(other) => {
                            return Err(ApiError::bad_request(
                                "unsupported_content",
                                format!("content part {other:?} is not supported"),
                            ));
                        }
                        None => {
                            return Err(ApiError::bad_request(
                                "invalid_content",
                                "content part is missing type",
                            ));
                        }
                    }
                }
                Ok(content)
            }
        }
    }
}

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

fn decode_image_part(part: &serde_json::Value) -> Result<Vec<u8>, ApiError> {
    let image = part.get("image_url").ok_or_else(|| {
        ApiError::bad_request("invalid_content", "image_url part has no image_url")
    })?;
    let url = image
        .as_str()
        .or_else(|| image.get("url").and_then(serde_json::Value::as_str))
        .ok_or_else(|| ApiError::bad_request("invalid_content", "image_url has no string url"))?;
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(ApiError::bad_request(
            "unsupported_image_url",
            "only data: image URLs are supported",
        ));
    };
    let Some((meta, payload)) = rest.split_once(',') else {
        return Err(ApiError::bad_request(
            "invalid_image_url",
            "data URL is missing a comma",
        ));
    };
    if !meta.ends_with(";base64") {
        return Err(ApiError::bad_request(
            "unsupported_image_url",
            "data image must be base64 encoded",
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|_| ApiError::bad_request("invalid_image", "image base64 is invalid"))?;
    if bytes.is_empty() {
        return Err(ApiError::bad_request("invalid_image", "image is empty"));
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(ApiError::bad_request(
            "image_too_large",
            format!("image is {} bytes, limit is {MAX_IMAGE_BYTES}", bytes.len()),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(value: serde_json::Value) -> ChatContent {
        serde_json::from_value(value).expect("content")
    }

    #[test]
    fn data_image_is_decoded_and_marker_inserted() {
        let content = parts(serde_json::json!([
            {"type": "text", "text": "look "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": " now"}
        ]))
        .into_parts("<image>")
        .expect("parse");
        assert_eq!(content.text, "look <image> now");
        assert_eq!(content.images, vec![b"hi".to_vec()]);
    }

    #[test]
    fn multiple_data_images_are_collected_with_markers() {
        let content = parts(serde_json::json!([
            {"type": "text", "text": "compare "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": " and "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,enk="}}
        ]))
        .into_parts("<image>")
        .expect("parse");
        assert_eq!(content.text, "compare <image> and <image>");
        assert_eq!(content.images, vec![b"hi".to_vec(), b"zy".to_vec()]);
    }

    #[test]
    fn remote_image_url_is_rejected() {
        let error = parts(serde_json::json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        ]))
        .into_parts("<image>")
        .expect_err("must reject");
        assert_eq!(error.code, "unsupported_image_url");
    }

    #[test]
    fn invalid_base64_is_rejected() {
        let error = parts(serde_json::json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}}
        ]))
        .into_parts("<image>")
        .expect_err("must reject");
        assert_eq!(error.code, "invalid_image");
    }
}
