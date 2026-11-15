use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{InspectError, Result};

pub(crate) fn choose_primary(files: &[PathBuf], preferred: &str) -> Vec<PathBuf> {
    if files.is_empty() {
        return Vec::new();
    }
    let mut files = files.to_vec();
    files.sort_by_key(|path| {
        let name = file_name(path);
        if name == preferred {
            0
        } else if name.contains("model") {
            1
        } else {
            2
        }
    });
    files
}

pub(crate) fn with_metadata(
    mut files: Vec<PathBuf>,
    tokenizer: Option<&Path>,
    config: Option<&Path>,
) -> Vec<PathBuf> {
    if let Some(tokenizer) = tokenizer {
        files.push(tokenizer.to_path_buf());
    }
    if let Some(config) = config {
        files.push(config.to_path_buf());
    }
    files
}

pub(crate) fn collect_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) -> Result<()> {
    if depth > 4 {
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|source| InspectError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| InspectError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let name = file_name(&path);
        if name.starts_with('.')
            || matches!(name.as_str(), "__pycache__" | "target" | "onnxruntime")
        {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, out, depth + 1)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) fn model_id(root: &Path) -> String {
    let name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    let mut id = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    if id.is_empty() {
        id = "model".into();
    }
    id
}

pub(crate) fn looks_like_whisper(root: &Path) -> bool {
    root.components()
        .filter_map(|component| component.as_os_str().to_str())
        .any(|component| component.to_ascii_lowercase().contains("whisper"))
}

pub(crate) fn extension(path: &Path) -> Option<&str> {
    path.extension().and_then(|extension| extension.to_str())
}

pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

pub(crate) fn relative(root: &Path, path: &Path) -> PathBuf {
    let relative = path
        .strip_prefix(root)
        .map_or_else(|_| path.to_path_buf(), Path::to_path_buf);
    let normalized = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    PathBuf::from(normalized)
}

pub(crate) fn relative_string(root: &Path, path: &Path) -> String {
    relative(root, path).to_string_lossy().into_owned()
}

pub(crate) fn sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}
