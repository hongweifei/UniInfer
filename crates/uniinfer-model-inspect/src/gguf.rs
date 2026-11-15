//! Minimal GGUF metadata reader.
//!
//! Reads the header and metadata key/value section only; tensor payloads are
//! never loaded. This gives architecture, quantization, context and tokenizer
//! vocabulary hints without a heavyweight GGML dependency.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use uniinfer_core::Quantization;

use crate::error::{InspectError, Result};

#[derive(Debug, Default)]
pub(crate) struct GgufMetadata {
    pub(crate) architecture: Option<String>,
    pub(crate) context_len: Option<usize>,
    pub(crate) quantization: Option<Quantization>,
    pub(crate) vocab_size: Option<usize>,
}

const MAX_STRING: u64 = 64 * 1024 * 1024;
const MAX_ARRAY: u64 = 10_000_000;

pub(crate) fn read(path: &Path) -> Result<GgufMetadata> {
    let file = File::open(path).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    read_exact(&mut reader, &mut magic, path)?;
    if &magic != b"GGUF" {
        return Err(InspectError::Gguf("invalid GGUF magic".into()));
    }
    let version = read_u32(&mut reader, path)?;
    if !(2..=3).contains(&version) {
        return Err(InspectError::Gguf(format!(
            "unsupported GGUF version {version}"
        )));
    }
    let _tensor_count = read_u64(&mut reader, path)?;
    let metadata_count = read_u64(&mut reader, path)?;
    let mut metadata = GgufMetadata::default();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader, path)?;
        let value_type = read_u32(&mut reader, path)?;
        match key.as_str() {
            "general.architecture" => {
                if value_type == 8 {
                    metadata.architecture = Some(read_string(&mut reader, path)?);
                } else {
                    skip_value(&mut reader, value_type, path)?;
                }
            }
            "general.file_type" => {
                if let Some(value) = read_unsigned(&mut reader, value_type, path)? {
                    metadata.quantization = file_type_quantization(value);
                } else {
                    skip_value(&mut reader, value_type, path)?;
                }
            }
            "general.context_length" => {
                if let Some(value) = read_unsigned(&mut reader, value_type, path)? {
                    metadata.context_len = usize::try_from(value).ok();
                } else {
                    skip_value(&mut reader, value_type, path)?;
                }
            }
            "tokenizer.ggml.tokens" => {
                if value_type == 9 {
                    let (element_type, count) = read_array_header(&mut reader, path)?;
                    metadata.vocab_size = usize::try_from(count).ok();
                    skip_array(&mut reader, element_type, count, path)?;
                } else {
                    skip_value(&mut reader, value_type, path)?;
                }
            }
            key if key.ends_with(".context_length") => {
                if let Some(value) = read_unsigned(&mut reader, value_type, path)? {
                    metadata.context_len = usize::try_from(value).ok();
                } else {
                    skip_value(&mut reader, value_type, path)?;
                }
            }
            _ => skip_value(&mut reader, value_type, path)?,
        }
    }
    Ok(metadata)
}

fn read_unsigned<R: Read>(reader: &mut R, value_type: u32, path: &Path) -> Result<Option<u64>> {
    match value_type {
        0 => Ok(Some(u64::from(read_u8(reader, path)?))),
        2 => Ok(Some(u64::from(read_u16(reader, path)?))),
        4 => Ok(Some(u64::from(read_u32(reader, path)?))),
        10 => Ok(Some(read_u64(reader, path)?)),
        _ => Ok(None),
    }
}

fn skip_value<R: Read>(reader: &mut R, value_type: u32, path: &Path) -> Result<()> {
    match value_type {
        0 | 1 => skip_bytes(reader, 1, path),
        2 | 3 => skip_bytes(reader, 2, path),
        4..=6 => skip_bytes(reader, 4, path),
        7 => skip_bytes(reader, 1, path),
        8 => {
            let length = read_u64(reader, path)?;
            skip_bytes(reader, length, path)
        }
        9 => {
            let (element_type, count) = read_array_header(reader, path)?;
            skip_array(reader, element_type, count, path)
        }
        10..=12 => skip_bytes(reader, 8, path),
        other => Err(InspectError::Gguf(format!(
            "unsupported GGUF metadata type {other}"
        ))),
    }
}

fn skip_array<R: Read>(reader: &mut R, element_type: u32, count: u64, path: &Path) -> Result<()> {
    if count > MAX_ARRAY {
        return Err(InspectError::Gguf(format!(
            "GGUF array is too large: {count}"
        )));
    }
    match element_type {
        8 => {
            for _ in 0..count {
                let length = read_u64(reader, path)?;
                skip_bytes(reader, length, path)?;
            }
            Ok(())
        }
        0 | 1 | 7 => skip_bytes(reader, count, path),
        2 | 3 => skip_bytes(reader, count * 2, path),
        4..=6 => skip_bytes(reader, count * 4, path),
        10..=12 => skip_bytes(reader, count * 8, path),
        other => Err(InspectError::Gguf(format!(
            "unsupported GGUF array type {other}"
        ))),
    }
}

fn read_array_header<R: Read>(reader: &mut R, path: &Path) -> Result<(u32, u64)> {
    let element_type = read_u32(reader, path)?;
    let count = read_u64(reader, path)?;
    Ok((element_type, count))
}

fn read_string<R: Read>(reader: &mut R, path: &Path) -> Result<String> {
    let length = read_u64(reader, path)?;
    if length > MAX_STRING {
        return Err(InspectError::Gguf(format!(
            "GGUF string is too large: {length}"
        )));
    }
    let mut bytes = vec![0u8; usize::try_from(length).unwrap_or(0)];
    read_exact(reader, &mut bytes, path)?;
    String::from_utf8(bytes).map_err(|error| InspectError::Gguf(format!("GGUF string: {error}")))
}

fn read_u8<R: Read>(reader: &mut R, path: &Path) -> Result<u8> {
    let mut bytes = [0u8; 1];
    read_exact(reader, &mut bytes, path)?;
    Ok(bytes[0])
}

fn read_u16<R: Read>(reader: &mut R, path: &Path) -> Result<u16> {
    let mut bytes = [0u8; 2];
    read_exact(reader, &mut bytes, path)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32<R: Read>(reader: &mut R, path: &Path) -> Result<u32> {
    let mut bytes = [0u8; 4];
    read_exact(reader, &mut bytes, path)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R, path: &Path) -> Result<u64> {
    let mut bytes = [0u8; 8];
    read_exact(reader, &mut bytes, path)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact<R: Read>(reader: &mut R, bytes: &mut [u8], path: &Path) -> Result<()> {
    reader.read_exact(bytes).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn skip_bytes<R: Read>(reader: &mut R, count: u64, path: &Path) -> Result<()> {
    let mut remaining = count;
    let mut buffer = [0u8; 8192];
    while remaining > 0 {
        let chunk = remaining.min(buffer.len() as u64) as usize;
        read_exact(reader, &mut buffer[..chunk], path)?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn file_type_quantization(file_type: u64) -> Option<Quantization> {
    match file_type {
        0 => Some(Quantization::Fp32),
        1 => Some(Quantization::Fp16),
        2 => Some(Quantization::Q4_0),
        3 => Some(Quantization::Q4_1),
        7 => Some(Quantization::Q8_0),
        8 => Some(Quantization::Q5_0),
        9 => Some(Quantization::Q5_1),
        10 => Some(Quantization::Q2K),
        11 => Some(Quantization::Q3KS),
        12 => Some(Quantization::Q3KM),
        13 => Some(Quantization::Q3KL),
        14 => Some(Quantization::Q4KS),
        15 => Some(Quantization::Q4KM),
        16 => Some(Quantization::Q5KS),
        17 => Some(Quantization::Q5KM),
        18 => Some(Quantization::Q6K),
        19 => Some(Quantization::Iq2Xxs),
        20 => Some(Quantization::Iq2Xs),
        21 => Some(Quantization::Q2KS),
        22 => Some(Quantization::Iq3Xs),
        23 => Some(Quantization::Iq3Xxs),
        24 => Some(Quantization::Iq1S),
        25 => Some(Quantization::Iq4Nl),
        26 => Some(Quantization::Iq3S),
        27 => Some(Quantization::Iq3M),
        28 => Some(Quantization::Iq2S),
        29 => Some(Quantization::Iq2M),
        30 => Some(Quantization::Iq4Xs),
        31 => Some(Quantization::Iq1M),
        32 => Some(Quantization::Bf16),
        36 => Some(Quantization::Tq1_0),
        37 => Some(Quantization::Tq2_0),
        38 => Some(Quantization::Mxfp4),
        39 => Some(Quantization::Nvfp4),
        40 => Some(Quantization::Q1_0),
        41 => Some(Quantization::Q2_0),
        _ => None,
    }
}
