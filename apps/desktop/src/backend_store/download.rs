//! HTTP transfer, archive extraction and integrity checks for engine distributions.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use reqwest::header::{ACCEPT_ENCODING, RANGE};
use tokio::io::AsyncReadExt;
use sha2::{Digest, Sha256};
use uniinfer_backend_store::{BackendFile, BackendManifest, BackendStore};

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
pub(crate) const CANCELLED: &str = "已取消";

pub(super) async fn download_package<P, C>(
    store: &BackendStore,
    package: &mut BackendManifest,
    staging: &Path,
    mut progress: P,
    cancel: C,
) -> Result<(), String>
where
    P: FnMut(u64, u64),
    C: Fn() -> bool,
{
    std::fs::create_dir_all(staging).map_err(|error| format!("创建暂存目录失败：{error}"))?;
    let total = package.files.iter().map(|file| file.size).sum::<u64>();
    let client = reqwest::Client::new();
    let mut done = 0_u64;
    for file in &package.files {
        if cancel() {
            return Err(CANCELLED.into());
        }
        if !BackendStore::safe_relative(&file.path) {
            return Err(format!("非法文件路径：{}", file.path));
        }
        if file.url.trim().is_empty() {
            return Err(format!("{} 没有下载地址", file.path));
        }
        let destination = staging.join(&file.path);
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| format!("创建目录失败：{error}"))?;
        }
        let before = done;
        done = done.saturating_add(
            download_file(
                &client,
                file,
                &destination,
                |bytes| progress(before.saturating_add(bytes), total),
                &cancel,
            )
            .await?,
        );
    }
    progress(done, total.max(done));
    extract_archives(staging, package)?;
    if !staging.join(&package.worker).is_file() {
        let wanted = Path::new(&package.worker)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("llama-server.exe");
        let Some(found) = find_file(staging, wanted) else {
            return Err(format!("引擎包缺少启动文件：{}", package.worker));
        };
        package.worker = found
            .strip_prefix(staging)
            .map_err(|error| format!("定位启动文件失败：{error}"))?
            .to_string_lossy()
            .into_owned();
    }
    store
        .write_manifest(staging, package)
        .map_err(|error| format!("写入引擎清单失败：{error}"))
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            return Some(path);
        }
    }
    None
}

fn extract_archives(staging: &Path, package: &BackendManifest) -> Result<(), String> {
    for file in &package.files {
        if !file.path.to_ascii_lowercase().ends_with(".zip") {
            continue;
        }
        let archive_path = staging.join(&file.path);
        let archive_file = std::fs::File::open(&archive_path)
            .map_err(|error| format!("打开 {} 失败：{error}", archive_path.display()))?;
        let mut archive = zip::ZipArchive::new(archive_file)
            .map_err(|error| format!("解压 {} 失败：{error}", archive_path.display()))?;
        archive
            .extract(staging)
            .map_err(|error| format!("解压 {} 失败：{error}", archive_path.display()))?;
        std::fs::remove_file(&archive_path)
            .map_err(|error| format!("清理 {} 失败：{error}", archive_path.display()))?;
    }
    Ok(())
}

async fn download_file<P, C>(
    client: &reqwest::Client,
    file: &BackendFile,
    destination: &Path,
    mut progress: P,
    cancel: C,
) -> Result<u64, String>
where
    P: FnMut(u64),
    C: Fn() -> bool,
{
    let tmp = destination.with_extension("part");
    let mut current = match tokio::fs::metadata(&tmp).await {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(format!("读取 {} 失败：{error}", tmp.display())),
    };
    if file.size > 0 && current > file.size {
        let _ = tokio::fs::remove_file(&tmp).await;
        current = 0;
    }
    let mut request = client
        .get(&file.url)
        .header(ACCEPT_ENCODING, "identity")
        .timeout(DOWNLOAD_TIMEOUT);
    if current > 0 {
        request = request.header(RANGE, format!("bytes={current}-"));
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("下载 {} 失败：{error}", file.path))?;
    let status = response.status();
    let resumed = current > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !resumed && !status.is_success() {
        return Err(format!("下载 {} 失败：HTTP {}", file.path, status));
    }
    if !resumed {
        current = 0;
    }
    let mut output = if resumed {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&tmp)
            .await
            .map_err(|error| format!("打开 {} 失败：{error}", tmp.display()))?
    } else {
        tokio::fs::File::create(&tmp)
            .await
            .map_err(|error| format!("创建 {} 失败：{error}", tmp.display()))?
    };
    let mut hasher = if file
        .sha256
        .as_deref()
        .is_some_and(|value| !value.is_empty())
    {
        Some(if resumed {
            hash_file(&tmp).await?
        } else {
            Sha256::new()
        })
    } else {
        None
    };
    let mut total = current;
    let mut last_progress = Instant::now();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if cancel() {
            return Err(CANCELLED.into());
        }
        let chunk = chunk.map_err(|error| format!("下载 {} 中断：{error}", file.path))?;
        total = total.saturating_add(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
        if let Some(hasher) = &mut hasher {
            hasher.update(&chunk);
        }
        tokio::io::AsyncWriteExt::write_all(&mut output, &chunk)
            .await
            .map_err(|error| format!("写入 {} 失败：{error}", tmp.display()))?;
        if last_progress.elapsed() >= Duration::from_millis(100) {
            progress(total);
            last_progress = Instant::now();
        }
    }
    tokio::io::AsyncWriteExt::flush(&mut output)
        .await
        .map_err(|error| format!("写入 {} 失败：{error}", tmp.display()))?;
    drop(output);
    if file.size > 0 && total != file.size {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(format!(
            "{} 大小不匹配：期望 {}，实际 {}",
            file.path, file.size, total
        ));
    }
    if let Some(expected) = file.sha256.as_deref().filter(|value| !value.is_empty()) {
        let actual = hex::encode(hasher.take().unwrap_or_default().finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(format!(
                "{} 校验失败：期望 {expected}，实际 {actual}",
                file.path
            ));
        }
    }
    tokio::fs::rename(&tmp, destination)
        .await
        .map_err(|error| format!("安装 {} 失败：{error}", destination.display()))?;
    progress(total);
    Ok(total)
}

async fn hash_file(path: &Path) -> Result<Sha256, String> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|error| format!("打开 {} 失败：{error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(|error| format!("读取 {} 失败：{error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher)
}
