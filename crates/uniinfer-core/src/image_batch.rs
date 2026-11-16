//! Encoded image batch carried through a single `image` tensor.
//!
//! Engine sessions receive named tensors, so multiple images share one `U8`
//! tensor. The envelope is length-prefixed and self-describing; a raw payload
//! without the magic prefix is treated as one image for direct session calls.

use crate::error::{EngineError, Result};

const MAGIC: &[u8; 4] = b"UIIM";
const MAX_IMAGES: usize = 256;

/// Encode images as `[magic][count][length, bytes]*`.
#[must_use]
pub fn encode_images(images: &[Vec<u8>]) -> Vec<u8> {
    let mut encoded =
        Vec::with_capacity(4 + 8 + images.iter().map(|image| 8 + image.len()).sum::<usize>());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&(images.len() as u64).to_le_bytes());
    for image in images {
        encoded.extend_from_slice(&(image.len() as u64).to_le_bytes());
        encoded.extend_from_slice(image);
    }
    encoded
}

/// Decode an encoded batch. Raw payloads decode as a single image.
pub fn decode_images(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    if !bytes.starts_with(MAGIC) {
        return Ok(vec![bytes.to_vec()]);
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
    Ok(images)
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
        let encoded = encode_images(&images);
        assert_eq!(decode_images(&encoded).expect("decode"), images);
    }

    #[test]
    fn raw_payload_decodes_as_one_image() {
        assert_eq!(
            decode_images(b"\x89PNG").expect("decode"),
            vec![b"\x89PNG".to_vec()]
        );
    }

    #[test]
    fn truncated_batch_is_rejected() {
        let mut encoded = encode_images(&[b"image".to_vec()]);
        encoded.pop();
        assert!(decode_images(&encoded).is_err());
    }
}
