//! Download, verification and installation of official inference engine distributions.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use uniinfer_backend_store::{BackendCatalog, BackendManifest, BackendStore, validate_manifest};

mod download;

use download::download_package;

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
        .timeout(Duration::from_secs(90))
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
pub(crate) async fn install_backend(package: BackendManifest) -> Result<InstallResult, String> {
    install_package(package).await
}

async fn install_package(package: BackendManifest) -> Result<InstallResult, String> {
    validate_manifest(&package).map_err(|error| error.to_string())?;
    if package.files.is_empty() {
        return Err("后端包没有文件清单".into());
    }
    let store = BackendStore::new(backend_dir());
    store.ensure().map_err(|error| error.to_string())?;
    let staging = staging_dir(&store, &package.id, &package.version);
    let result = download_package(&store, &package, &staging).await;
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    finalize_package(&store, &package, &staging)
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
        std::fs::rename(&destination, &old)
            .map_err(|error| format!("替换旧后端失败：{}", error))?;
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
