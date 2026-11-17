//! Resumable HTTP transfer, checksum verification and local path safety.

use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::StatusCode;
use reqwest::header::{ACCEPT_ENCODING, RANGE};
use sha2::{Digest, Sha256};
use tauri::AppHandle;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::super::RemoteFile;
use super::{DownloadManager, emit_progress};

const DOWNLOAD_ATTEMPTS: usize = 3;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(120);
const READ_POLL: Duration = Duration::from_millis(500);
const READ_IDLE: Duration = Duration::from_secs(60);

pub(super) async fn download_file(
    app: Option<&AppHandle>,
    manager: &DownloadManager,
    task_id: &str,
    client: &reqwest::Client,
    file: &RemoteFile,
    destination: &Path,
) -> Result<u64, InstallError> {
    let done_before = manager
        .get(task_id)
        .map(|task| task.downloaded)
        .unwrap_or(0);
    let total = manager.get(task_id).map(|task| task.total).unwrap_or(0);
    let part = part_path(destination);
    if destination.is_file() {
        let length = tokio::fs::metadata(destination)
            .await
            .map_err(|error| {
                InstallError::Message(format!("stat {}: {error}", destination.display()))
            })?
            .len();
        if file.size == 0 || length == file.size {
            return Ok(length);
        }
    }
    let mut current = match tokio::fs::metadata(&part).await {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => {
            return Err(InstallError::Message(format!(
                "stat {}: {error}",
                part.display()
            )));
        }
    };
    if file.size > 0 && current > file.size {
        tokio::fs::remove_file(&part).await.map_err(|error| {
            InstallError::Message(format!("remove {}: {error}", part.display()))
        })?;
        current = 0;
    }
    if file.size > 0 && current == file.size && verify_part(&part, file).await? {
        rename_part(&part, destination).await?;
        emit_progress(app, manager, task_id, file, done_before + current, total);
        return Ok(current);
    }
    if file.size > 0 && current == file.size {
        let _ = tokio::fs::remove_file(&part).await;
        current = 0;
    }

    let mut last_error = String::new();
    for attempt in 0..DOWNLOAD_ATTEMPTS {
        check_cancelled(manager, task_id)?;
        let mut request = client.get(&file.url).header(ACCEPT_ENCODING, "identity");
        if current > 0 {
            request = request.header(RANGE, format!("bytes={current}-"));
        }
        let response = match tokio::time::timeout(Duration::from_secs(60), request.send()).await {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                last_error = format!("download {}: {error}", file.url);
                retry_or_break(attempt, &mut last_error).await?;
                continue;
            }
            Err(_) => {
                last_error = format!("download {}: 请求超时", file.url);
                retry_or_break(attempt, &mut last_error).await?;
                continue;
            }
        };
        let status = response.status();
        let resumed = current > 0 && status == StatusCode::PARTIAL_CONTENT;
        if !resumed && status != StatusCode::OK {
            last_error = format!("download {}: HTTP {status}", file.url);
            if status.is_client_error() && status != StatusCode::REQUEST_TIMEOUT {
                break;
            }
            retry_or_break(attempt, &mut last_error).await?;
            continue;
        }
        if !resumed {
            current = 0;
        }
        let mut output = if resumed {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&part)
                .await
                .map_err(|error| {
                    InstallError::Message(format!("open {}: {error}", part.display()))
                })?
        } else {
            tokio::fs::File::create(&part).await.map_err(|error| {
                InstallError::Message(format!("create {}: {error}", part.display()))
            })?
        };
        let mut hasher = if file.sha256.is_some() {
            Some(if resumed {
                hash_existing(&part).await?
            } else {
                Sha256::new()
            })
        } else {
            None
        };
        let mut stream = response.bytes_stream();
        let mut last_emit = Instant::now();
        let mut last_data = Instant::now();
        let mut stream_error = None;
        emit_progress(app, manager, task_id, file, done_before + current, total);
        loop {
            check_cancelled(manager, task_id)?;
            let next = tokio::time::timeout(READ_POLL, stream.next()).await;
            match next {
                Err(_) => {
                    if last_data.elapsed() >= READ_IDLE {
                        stream_error = Some(format!("download {}: 读取超时", file.url));
                        break;
                    }
                    continue;
                }
                Ok(Some(Ok(chunk))) => {
                    last_data = Instant::now();
                    current += chunk.len() as u64;
                    if let Some(hasher) = &mut hasher {
                        hasher.update(&chunk);
                    }
                    output.write_all(&chunk).await.map_err(|error| {
                        InstallError::Message(format!("write {}: {error}", part.display()))
                    })?;
                    if last_emit.elapsed() >= PROGRESS_INTERVAL {
                        emit_progress(app, manager, task_id, file, done_before + current, total);
                        last_emit = Instant::now();
                    }
                }
                Ok(Some(Err(error))) => {
                    stream_error = Some(format!("download {}: {error}", file.url));
                    break;
                }
                Ok(None) => break,
            }
        }
        output
            .flush()
            .await
            .map_err(|error| InstallError::Message(format!("flush {}: {error}", part.display())))?;
        drop(output);
        if let Some(error) = stream_error {
            last_error = error;
            retry_or_break(attempt, &mut last_error).await?;
            continue;
        }
        if file.size > 0 && current != file.size {
            last_error = format!(
                "download {}: expected {} bytes, got {current}",
                file.url, file.size
            );
            retry_or_break(attempt, &mut last_error).await?;
            continue;
        }
        if let Some(expected) = &file.sha256 {
            let actual = hex::encode(hasher.take().unwrap_or_default().finalize());
            if !actual.eq_ignore_ascii_case(expected) {
                let _ = tokio::fs::remove_file(&part).await;
                current = 0;
                last_error = format!(
                    "checksum mismatch for {}: expected {expected}, got {actual}",
                    file.path
                );
                retry_or_break(attempt, &mut last_error).await?;
                continue;
            }
        }
        rename_part(&part, destination).await?;
        emit_progress(app, manager, task_id, file, done_before + current, total);
        return Ok(current);
    }
    Err(InstallError::Message(last_error))
}

async fn retry_or_break(attempt: usize, last_error: &mut String) -> Result<(), InstallError> {
    if attempt + 1 >= DOWNLOAD_ATTEMPTS {
        return Err(InstallError::Message(std::mem::take(last_error)));
    }
    tokio::time::sleep(Duration::from_millis(400 * (attempt as u64 + 1))).await;
    Ok(())
}

async fn verify_part(path: &Path, file: &RemoteFile) -> Result<bool, InstallError> {
    let Some(expected) = &file.sha256 else {
        return Ok(true);
    };
    let actual = hex::encode(hash_existing(path).await?.finalize());
    Ok(actual.eq_ignore_ascii_case(expected))
}

async fn hash_existing(path: &Path) -> Result<Sha256, InstallError> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|error| InstallError::Message(format!("open {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(|error| InstallError::Message(format!("read {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher)
}

async fn rename_part(part: &Path, destination: &Path) -> Result<(), InstallError> {
    tokio::fs::rename(part, destination).await.map_err(|error| {
        InstallError::Message(format!(
            "rename {} -> {}: {error}",
            part.display(),
            destination.display()
        ))
    })
}

pub(super) fn part_path(destination: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "download".into());
    destination.with_file_name(format!("{name}.part"))
}

pub(super) fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, InstallError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(InstallError::Message(format!(
            "模型文件路径不安全：{relative}"
        )));
    }
    Ok(root.join(path))
}

pub(super) fn check_cancelled(manager: &DownloadManager, id: &str) -> Result<(), InstallError> {
    if manager.is_cancelled(id) {
        Err(InstallError::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub(super) enum InstallError {
    Cancelled,
    Message(String),
}

impl From<String> for InstallError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::Source;

    #[test]
    fn rejects_paths_that_escape_the_staging_directory() {
        let root = Path::new("staging");
        assert!(safe_join(root, "model.gguf").is_ok());
        assert!(safe_join(root, "sub/model.gguf").is_ok());
        assert!(safe_join(root, "../model.gguf").is_err());
        assert!(safe_join(root, "/tmp/model.gguf").is_err());
        assert!(safe_join(root, "sub/../model.gguf").is_err());
    }

    #[test]
    fn keeps_part_files_next_to_their_destination() {
        let part = part_path(Path::new("models/model.gguf"));
        assert_eq!(part, PathBuf::from("models/model.gguf.part"));
    }

    #[tokio::test]
    async fn resumes_partial_download_and_verifies_checksum() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let payload = b"UniInfer downloadable payload".repeat(4096);
        let total = payload.len();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let range_seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let range_flag = range_seen.clone();
        let body = payload.clone();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept");
            let mut request = [0u8; 4096];
            let read = socket.read(&mut request).expect("read request");
            let request = String::from_utf8_lossy(&request[..read]);
            let start = request
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("range: bytes=")
                        .and_then(|value| value.split('-').next())
                        .and_then(|value| value.parse::<usize>().ok())
                })
                .unwrap_or(0);
            if start > 0 {
                range_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            let body = &body[start..];
            let header = if start > 0 {
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    body.len(),
                    start,
                    total - 1,
                    total
                )
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
            };
            socket.write_all(header.as_bytes()).expect("headers");
            socket.write_all(body).expect("body");
        });

        let directory = tempfile::tempdir().expect("tempdir");
        let destination = directory.path().join("artifact.bin");
        let partial = part_path(&destination);
        tokio::fs::write(&partial, &payload[..17])
            .await
            .expect("partial");
        let manager = DownloadManager::new();
        let task = manager
            .enqueue("test/model".into(), Source::HuggingFace, None)
            .expect("task");
        let file = RemoteFile {
            path: "artifact.bin".into(),
            url: format!("http://{address}/artifact.bin"),
            sha256: Some(hex::encode(Sha256::digest(&payload))),
            size: payload.len() as u64,
        };
        let client = reqwest::Client::new();
        let downloaded = download_file(None, &manager, &task.id, &client, &file, &destination)
            .await
            .expect("download");
        assert_eq!(downloaded, payload.len() as u64);
        assert_eq!(tokio::fs::read(&destination).await.expect("file"), payload);
        assert!(!partial.exists());
        assert!(range_seen.load(std::sync::atomic::Ordering::Relaxed));
        server.join().expect("server");
    }
}
