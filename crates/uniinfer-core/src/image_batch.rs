//! Ordered image batch carried through a single `image` tensor.
//!
//! Engine sessions exchange named tensors, so multiple images share one `U8`
//! tensor. The payload is always a `UIIM` envelope; there is no implicit raw
//! single-image form, which keeps API, ONNX, Candle and llama.cpp on one
//! contract.

use crate::error::{EngineError, Result};

const MAGIC: &[u8; 4] = b"UIIM";
/// Upper bound for one request across every engine.
pub const MAX_IMAGES: usize = 256;

/// Decoded images in request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBatch {
    images: Vec<Vec<u8>>,
}

impl ImageBatch {
    /// Encode images as `[magic][count][length, bytes]*`.
    pub fn encode(images: &[Vec<u8>]) -> Result<Vec<u8>> {
        if images.is_empty() {
            return Err(EngineError::Other("image batch is empty".into()));
        }
        if images.len() > MAX_IMAGES {
            return Err(EngineError::Other(format!(
                "image batch count {} exceeds {MAX_IMAGES}",
                images.len()
            )));
        }
        let mut encoded =
            Vec::with_capacity(4 + 8 + images.iter().map(|image| 8 + image.len()).sum::<usize>());
        encoded.extend_from_slice(MAGIC);
        encoded.extend_from_slice(&(images.len() as u64).to_le_bytes());
        for image in images {
            encoded.extend_from_slice(&(image.len() as u64).to_le_bytes());
            encoded.extend_from_slice(image);
        }
        Ok(encoded)
    }

    /// Decode a `UIIM` envelope.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if !bytes.starts_with(MAGIC) {
            return Err(EngineError::Other(
                "image tensor is missing the UIIM header".into(),
            ));
        }
        let mut cursor = MAGIC.len();
        let count = read_u64(bytes, &mut cursor)?;
        let count = usize::try_from(count)
            .map_err(|_| EngineError::Other("image batch count overflow".into()))?;
        if count == 0 || count > MAX_IMAGES {
            return Err(EngineError::Other(format!(
                "image batch count {count} is outside 1..={MAX_IMAGES}"
            )));
        }
        let mut images = Vec::with_capacity(count);
        for _ in 0..count {
            let len = usize::try_from(read_u64(bytes, &mut cursor)?)
                .map_err(|_| EngineError::Other("image batch length overflow".into()))?;
            let end = cursor
                .checked_add(len)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| EngineError::Other("image batch is truncated".into()))?;
            images.push(bytes[cursor..end].to_vec());
            cursor = end;
        }
        if cursor != bytes.len() {
            return Err(EngineError::Other("image batch has trailing bytes".into()));
        }
        Ok(Self { images })
    }

    #[must_use]
    pub fn as_slice(&self) -> &[Vec<u8>] {
        &self.images
    }
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = cursor
        .checked_add(8)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| EngineError::Other("image batch header is truncated".into()))?;
    let value = u64::from_le_bytes(
        bytes[*cursor..end]
            .try_into()
            .map_err(|_| EngineError::Other("image batch header is truncated".into()))?,
    );
    *cursor = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_multiple_images() {
        let images = vec![b"first".to_vec(), b"second".to_vec()];
        let encoded = ImageBatch::encode(&images).expect("encode");
        assert_eq!(
            ImageBatch::decode(&encoded).expect("decode").as_slice(),
            images
        );
    }

    #[test]
    fn empty_batch_is_rejected() {
        assert!(ImageBatch::encode(&[]).is_err());
    }

    #[test]
    fn too_many_images_are_rejected() {
        let images = vec![Vec::new(); MAX_IMAGES + 1];
        assert!(ImageBatch::encode(&images).is_err());
    }

    #[test]
    fn raw_payload_is_rejected() {
        assert!(ImageBatch::decode(b"\x89PNG").is_err());
    }

    #[test]
    fn truncated_batch_is_rejected() {
        let mut encoded = ImageBatch::encode(&[b"image".to_vec()]).expect("encode");
        encoded.pop();
        assert!(ImageBatch::decode(&encoded).is_err());
    }
}
