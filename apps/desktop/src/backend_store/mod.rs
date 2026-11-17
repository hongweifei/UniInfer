//! Download, verification and installation of official inference engine distributions.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use uniinfer_backend_store::{
    BackendCatalog, BackendEngine, BackendFile, BackendLauncher, BackendManifest, BackendStore,
    validate_manifest,
};

mod download;

use download::download_package;

#[derive(Clone, serde::Serialize)]
pub(crate) struct InstallResult {
    id: String,
    version: String,
    path: PathBuf,
    restart_required: bool,
}

const LLAMA_RELEASE_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

#[derive(serde::Deserialize)]
struct GithubRelease {
    tag_name: String,
    html_url: String,
    assets: Vec<GithubAsset>,
}

#[derive(Clone, serde::Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

/// Read the official llama.cpp release feed and project its Windows assets into
/// UniInfer engine variants. No user-maintained catalog URL is involved.
#[tauri::command]
pub(crate) async fn fetch_backend_catalog() -> Result<BackendCatalog, String> {
    let client = reqwest::Client::new();
    let response = client
        .get(LLAMA_RELEASE_API)
        .timeout(Duration::from_secs(90))
        .header(reqwest::header::USER_AGENT, "UniInfer")
        .send()
        .await
        .map_err(|error| format!("获取 llama.cpp 官方版本失败：{error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "获取 llama.cpp 官方版本失败：HTTP {}",
            response.status()
        ));
    }
    let release: GithubRelease = response
        .json()
        .await
        .map_err(|error| format!("解析 llama.cpp 官方版本失败：{error}"))?;
    let mut catalog = BackendCatalog::default();
    for (variant, label, devices) in [
        ("Vulkan", "Vulkan", vec![uniinfer_core::DeviceClass::Gpu]),
        ("CUDA", "CUDA", vec![uniinfer_core::DeviceClass::Gpu]),
        ("CPU", "CPU", vec![uniinfer_core::DeviceClass::Cpu]),
    ] {
        let Some((asset, cudart)) = find_windows_assets(&release.assets, variant) else {
            continue;
        };
        let id = format!("llama-cpp-{}", variant.to_ascii_lowercase());
        let mut files = vec![BackendFile {
            path: asset.name.clone(),
            url: asset.browser_download_url.clone(),
            sha256: None,
            size: asset.size,
        }];
        if let Some(cudart) = cudart {
            files.push(BackendFile {
                path: cudart.name.clone(),
                url: cudart.browser_download_url.clone(),
                sha256: None,
                size: cudart.size,
            });
        }
        catalog.backends.push(BackendManifest {
            id: id.clone(),
            name: "llama.cpp".into(),
            version: release.tag_name.clone(),
            description: format!("官方 llama.cpp Windows {label} 构建"),
            homepage: Some(release.html_url.clone()),
            license: Some("MIT".into()),
            platform: "windows-x86_64".into(),
            worker: "llama-server.exe".into(),
            launcher: BackendLauncher::LlamaServer,
            devices: devices.clone(),
            engines: vec![BackendEngine {
                id: id.clone(),
                name: "llama.cpp".into(),
                runtime: "llama".into(),
                description: format!("官方 llama.cpp {label} 构建"),
                vendor: "llama.cpp".into(),
                variant: label.into(),
                category: "language".into(),
                homepage: Some(release.html_url.clone()),
                implementations: vec!["llama.text".into()],
                formats: vec!["gguf".into()],
                modalities: vec!["text".into()],
                devices,
            }],
            files,
        });
    }
    if catalog.backends.is_empty() {
        return Err("llama.cpp 官方版本没有可用的 Windows 引擎资产".into());
    }
    for package in &catalog.backends {
        validate_manifest(package).map_err(|error| error.to_string())?;
    }
    Ok(catalog)
}

fn find_windows_assets<'a>(
    assets: &'a [GithubAsset],
    variant: &str,
) -> Option<(&'a GithubAsset, Option<&'a GithubAsset>)> {
    let needle = variant.to_ascii_lowercase();
    let asset = assets.iter().find(|asset| {
        let name = asset.name.to_ascii_lowercase();
        name.ends_with(".zip")
            && name.contains("bin-win")
            && name.contains(&needle)
            && !name.contains("cudart")
            && !name.contains("arm64")
    })?;
    let cudart = (variant == "CUDA")
        .then(|| {
            assets.iter().find(|asset| {
                let name = asset.name.to_ascii_lowercase();
                name.ends_with(".zip") && name.contains("cudart") && name.contains("cuda")
            })
        })
        .flatten();
    Some((asset, cudart))
}

#[tauri::command]
pub(crate) async fn install_backend(package: BackendManifest) -> Result<InstallResult, String> {
    install_package(package).await
}

async fn install_package(mut package: BackendManifest) -> Result<InstallResult, String> {
    validate_manifest(&package).map_err(|error| error.to_string())?;
    if package.files.is_empty() {
        return Err("后端包没有文件清单".into());
    }
    let store = BackendStore::new(backend_dir());
    store.ensure().map_err(|error| error.to_string())?;
    let staging = staging_dir(&store, &package.id, &package.version);
    let result = download_package(&store, &mut package, &staging).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> GithubAsset {
        GithubAsset {
            name: name.into(),
            browser_download_url: format!("https://example.com/{name}"),
            size: 1,
        }
    }

    #[test]
    fn selects_official_windows_variants() {
        let assets = vec![
            asset("llama-b1-bin-win-vulkan-x64.zip"),
            asset("llama-b1-bin-win-cuda-12.4-x64.zip"),
            asset("cudart-llama-bin-win-cu12.4-x64.zip"),
            asset("llama-b1-bin-win-cpu-x64.zip"),
            asset("llama-b1-bin-win-arm64.zip"),
        ];
        assert_eq!(
            find_windows_assets(&assets, "Vulkan").map(|(asset, _)| asset.name.as_str()),
            Some("llama-b1-bin-win-vulkan-x64.zip")
        );
        assert_eq!(
            find_windows_assets(&assets, "CUDA").map(|(asset, _)| asset.name.as_str()),
            Some("llama-b1-bin-win-cuda-12.4-x64.zip")
        );
        assert_eq!(
            find_windows_assets(&assets, "CPU").map(|(asset, _)| asset.name.as_str()),
            Some("llama-b1-bin-win-cpu-x64.zip")
        );
        assert!(find_windows_assets(&assets, "ROCm").is_none());
    }
}
