//! HTTP transfer, archive extraction and integrity checks for engine distributions.

use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::ACCEPT_ENCODING;
use sha2::{Digest, Sha256};
use uniinfer_backend_store::{BackendFile, BackendManifest, BackendStore};

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(90);

pub(super) async fn download_package(
    store: &BackendStore,
    package: &BackendManifest,
    staging: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(staging).map_err(|error| format!("创建暂存目录失败：{error}"))?;
    let client = reqwest::Client::new();
    for file in &package.files {
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
        download_file(&client, file, &destination).await?;
    }
    extract_archives(staging, package)?;
    if !staging.join(&package.worker).is_file() {
        return Err(format!("引擎包缺少启动文件：{}", package.worker));
    }
    store
        .write_manifest(staging, package)
        .map_err(|error| format!("写入引擎清单失败：{error}"))
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

async fn download_file(
    client: &reqwest::Client,
    file: &BackendFile,
    destination: &Path,
) -> Result<(), String> {
    let response = client
        .get(&file.url)
        .header(ACCEPT_ENCODING, "identity")
        .timeout(DOWNLOAD_TIMEOUT)
        .send()
        .await
        .map_err(|error| format!("下载 {} 失败：{error}", file.path))?;
    if !response.status().is_success() {
        return Err(format!(
            "下载 {} 失败：HTTP {}",
            file.path,
            response.status()
        ));
    }
    let tmp = destination.with_extension("part");
    let mut output = tokio::fs::File::create(&tmp)
        .await
        .map_err(|error| format!("创建 {} 失败：{error}", tmp.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("下载 {} 中断：{error}", file.path))?;
        total = total.saturating_add(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
        hasher.update(&chunk);
        tokio::io::AsyncWriteExt::write_all(&mut output, &chunk)
            .await
            .map_err(|error| format!("写入 {} 失败：{error}", tmp.display()))?;
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
        let actual = hex::encode(hasher.finalize());
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
        .map_err(|error| format!("安装 {} 失败：{error}", destination.display()))
}
