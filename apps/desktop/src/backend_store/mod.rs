//! Download, verification and installation of official inference engine distributions.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use uniinfer_backend_store::{BackendCatalog, BackendManifest, BackendStore, validate_manifest};

mod catalog;
pub(crate) mod download;

use download::download_package;

#[tauri::command]
pub(crate) fn import_backend_archive(path: String) -> Result<InstallResult, String> {
    let store = BackendStore::new(backend_dir());
    store.ensure().map_err(|error| error.to_string())?;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let staging = store.root().join(format!(".staging-import-{nanos}"));
    std::fs::create_dir_all(&staging).map_err(|error| format!("创建暂存目录失败：{error}"))?;
    let result = (|| {
        extract_import_zip(Path::new(&path), &staging)?;
        let manifest_path = staging.join("backend.toml");
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|error| format!("读取 backend.toml 失败：{error}"))?;
        let package: BackendManifest = toml::from_str(&raw)
            .map_err(|error| format!("解析 backend.toml 失败：{error}"))?;
        validate_manifest(&package).map_err(|error| error.to_string())?;
        if package.files.is_empty() {
            return Err("后端包没有文件清单".into());
        }
        if !staging.join(&package.worker).is_file() {
            return Err(format!("后端包缺少启动文件：{}", package.worker));
        }
        finalize_package(&store, &package, &staging)
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}

fn extract_import_zip(path: &Path, staging: &Path) -> Result<(), String> {
    let file = std::fs::File::open(path).map_err(|error| format!("打开 zip 失败：{error}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| format!("解析 zip 失败：{error}"))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("读取 zip 条目失败：{error}"))?;
        let Some(relative) = entry.enclosed_name() else {
            return Err("后端包包含非法路径".into());
        };
        let destination = staging.join(&relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&destination)
                .map_err(|error| format!("创建目录失败：{error}"))?;
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("创建目录失败：{error}"))?;
        }
        let mut output = std::fs::File::create(&destination)
            .map_err(|error| format!("创建 {} 失败：{error}", destination.display()))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|error| format!("写入 {} 失败：{error}", destination.display()))?;
    }
    Ok(())
}

#[tauri::command]
pub(crate) async fn fetch_backend_catalog() -> Result<BackendCatalog, String> {
    catalog::fetch().await
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct InstallResult {
    pub(crate) id: String,
    pub(crate) version: String,
    pub(crate) path: PathBuf,
}

pub(crate) async fn install_package_with<P, C>(
    mut package: BackendManifest,
    mut progress: P,
    cancel: C,
) -> Result<InstallResult, String>
where
    P: FnMut(u64, u64),
    C: Fn() -> bool,
{
    validate_manifest(&package).map_err(|error| error.to_string())?;
    if package.files.is_empty() {
        return Err("后端包没有文件清单".into());
    }
    let store = BackendStore::new(backend_dir());
    store.ensure().map_err(|error| error.to_string())?;
    let staging = staging_dir(&store, &package.id, &package.version);
    let result = download_package(&store, &mut package, &staging, &mut progress, &cancel).await;
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(error);
    }
    if cancel() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(download::CANCELLED.into());
    }
    progress(
        package.files.iter().map(|file| file.size).sum(),
        package.files.iter().map(|file| file.size).sum(),
    );
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
