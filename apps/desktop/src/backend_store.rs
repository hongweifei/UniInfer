//! Download and local import of sidecar inference backend packages.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use reqwest::header::ACCEPT_ENCODING;
use sha2::{Digest, Sha256};
use tauri::AppHandle;
use uniinfer_backend_store::{
    BackendCatalog, BackendFile, BackendManifest, BackendStore, validate_manifest,
};

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Clone, serde::Serialize)]
pub(crate) struct InstallResult {
    id: String,
    version: String,
    path: PathBuf,
    restart_required: bool,
}

#[tauri::command]
pub(crate) async fn fetch_backend_catalog(url: String) -> Result<BackendCatalog, String> {
    let client = reqwest::Client::new();
    let response = client
        .get(url.trim())
        .timeout(DOWNLOAD_TIMEOUT)
        .send()
        .await
        .map_err(|error| format!("获取后端目录失败：{error}"))?;
    if !response.status().is_success() {
        return Err(format!("获取后端目录失败：HTTP {}", response.status()));
    }
    let catalog: BackendCatalog = response
        .json()
        .await
        .map_err(|error| format!("解析后端目录失败：{error}"))?;
    for package in &catalog.backends {
        validate_manifest(package).map_err(|error| error.to_string())?;
    }
    Ok(catalog)
}

#[tauri::command]
pub(crate) async fn install_backend(
    app: AppHandle,
    package: BackendManifest,
) -> Result<InstallResult, String> {
    install_package(&app, package).await
}

async fn install_package(
    app: &AppHandle,
    package: BackendManifest,
) -> Result<InstallResult, String> {
    validate_manifest(&package).map_err(|error| error.to_string())?;
    if package.files.is_empty() {
        return Err("后端包没有文件清单".into());
    }
    let store = BackendStore::new(backend_dir());
    store.ensure().map_err(|error| error.to_string())?;
    let staging = staging_dir(&store, &package.id, &package.version);
    let result = download_package(app, &store, &package, &staging).await;
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    finalize_package(&store, &package, &staging)
}

async fn download_package(
    _app: &AppHandle,
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
        .map_err(|error| format!("写入后端清单失败：{error}"))
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
        return Err(format!("下载 {} 失败：HTTP {}", file.path, response.status()));
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

fn finalize_package(
    store: &BackendStore,
    package: &BackendManifest,
    staging: &Path,
) -> Result<InstallResult, String> {
    let destination = store.package_dir(&package.id, &package.version);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("创建目录失败：{error}"))?;
    }
    let old = destination.with_extension("old");
    if destination.exists() {
        std::fs::rename(&destination, &old).map_err(|error| {
            format!("替换旧后端失败：{}", error)
        })?;
    }
    if let Err(error) = std::fs::rename(staging, &destination) {
        if old.exists() {
            let _ = std::fs::rename(&old, &destination);
        }
        return Err(format!("安装后端失败：{error}"));
    }
    let _ = std::fs::remove_dir_all(&old);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let worker = destination.join(&package.worker);
        if let Ok(metadata) = std::fs::metadata(&worker) {
            let mut permissions = metadata.permissions();
            permissions.set_mode(0o755);
            let _ = std::fs::set_permissions(&worker, permissions);
        }
    }
    Ok(InstallResult {
        id: package.id.clone(),
        version: package.version.clone(),
        path: destination,
        restart_required: true,
    })
}

fn staging_dir(store: &BackendStore, id: &str, version: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    store
        .root()
        .join(format!(".staging-{id}-{version}-{nanos}"))
}

fn backend_dir() -> PathBuf {
    crate::api_server::backend_dir()
}
