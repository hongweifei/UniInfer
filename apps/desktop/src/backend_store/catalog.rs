//! Official GitHub release discovery for llama.cpp and stable-diffusion.cpp.

use std::time::Duration;

use uniinfer_backend_store::{
    BackendCatalog, BackendEngine, BackendFile, BackendLauncher, BackendManifest, validate_manifest,
};

const GITHUB_MIRROR: &str = "https://gh-proxy.com/";

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
    #[serde(default)]
    digest: Option<String>,
}

pub(super) async fn fetch() -> Result<BackendCatalog, String> {
    let client = reqwest::Client::new();
    let mut catalog = BackendCatalog::default();
    let mut errors = Vec::new();
    match github_release(&client, "ggml-org/llama.cpp").await {
        Ok((release, mirrored)) => catalog.backends.extend(llama_catalog(&release, mirrored)),
        Err(error) => errors.push(format!("llama.cpp: {error}")),
    }
    match github_release(&client, "leejet/stable-diffusion.cpp").await {
        Ok((release, mirrored)) => catalog.backends.extend(sd_catalog(&release, mirrored)),
        Err(error) => errors.push(format!("stable-diffusion.cpp: {error}")),
    }
    if catalog.backends.is_empty() {
        return Err(format!(
            "官方 release 没有可用的 Windows 推理引擎：{}",
            errors.join("；")
        ));
    }
    for package in &catalog.backends {
        validate_manifest(package).map_err(|error| error.to_string())?;
    }
    Ok(catalog)
}

async fn github_release(
    client: &reqwest::Client,
    repo: &str,
) -> Result<(GithubRelease, bool), String> {
    let direct = format!("https://api.github.com/repos/{repo}/releases/latest");
    match fetch_release(client, &direct).await {
        Ok(release) => Ok((release, false)),
        Err(direct_error) => {
            let mirrored = format!("{GITHUB_MIRROR}{direct}");
            fetch_release(client, &mirrored)
                .await
                .map(|release| (release, true))
                .map_err(|mirror_error| {
                    format!(
                        "无法访问 GitHub（直连：{direct_error}；镜像：{mirror_error}）。请检查网络或代理，也可以稍后重试。"
                    )
                })
        }
    }
}

async fn fetch_release(client: &reqwest::Client, url: &str) -> Result<GithubRelease, String> {
    let response = client
        .get(url)
        .timeout(Duration::from_secs(90))
        .header(reqwest::header::USER_AGENT, "UniInfer")
        .send()
        .await
        .map_err(|error| format!("{error}"))?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    response
        .json()
        .await
        .map_err(|error| format!("解析响应失败：{error}"))
}

fn asset_url(url: &str, mirrored: bool) -> String {
    if mirrored {
        format!("{GITHUB_MIRROR}{url}")
    } else {
        url.to_owned()
    }
}

fn asset_file(asset: &GithubAsset, mirrored: bool) -> BackendFile {
    BackendFile {
        path: asset.name.clone(),
        url: asset_url(&asset.browser_download_url, mirrored),
        sha256: asset
            .digest
            .as_deref()
            .and_then(|digest| digest.strip_prefix("sha256:"))
            .map(str::to_owned),
        size: asset.size,
    }
}

fn llama_catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    let mut backends = Vec::new();
    for (variant, label, devices) in [
        ("Vulkan", "Vulkan", vec![uniinfer_core::DeviceClass::Gpu]),
        ("CUDA", "CUDA", vec![uniinfer_core::DeviceClass::Gpu]),
        ("CPU", "CPU", vec![uniinfer_core::DeviceClass::Cpu]),
    ] {
        let Some(asset) = find_windows_asset(&release.assets, "llama", variant) else {
            continue;
        };
        let id = format!("llama-cpp-{}", variant.to_ascii_lowercase());
        let mut files = vec![asset_file(asset, mirrored)];
        if variant == "CUDA"
            && let Some(cudart) = find_windows_asset(&release.assets, "cudart", "cuda")
        {
            files.push(asset_file(cudart, mirrored));
        }
        backends.push(BackendManifest {
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
    backends
}

fn sd_catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    let mut backends = Vec::new();
    for (variant, label, devices) in [
        ("Vulkan", "Vulkan", vec![uniinfer_core::DeviceClass::Gpu]),
        ("CUDA", "CUDA", vec![uniinfer_core::DeviceClass::Gpu]),
        ("CPU", "CPU", vec![uniinfer_core::DeviceClass::Cpu]),
    ] {
        let Some(asset) = find_windows_asset(&release.assets, "sd-", variant) else {
            continue;
        };
        let id = format!("stable-diffusion-{}", variant.to_ascii_lowercase());
        let mut files = vec![asset_file(asset, mirrored)];
        if variant == "CUDA"
            && let Some(cudart) = find_windows_asset(&release.assets, "cudart", "cuda")
        {
            files.push(asset_file(cudart, mirrored));
        }
        backends.push(BackendManifest {
            id: id.clone(),
            name: "stable-diffusion.cpp".into(),
            version: release.tag_name.clone(),
            description: format!("官方 stable-diffusion.cpp Windows {label} 构建"),
            homepage: Some(release.html_url.clone()),
            license: Some("MIT".into()),
            platform: "windows-x86_64".into(),
            worker: "sd-server.exe".into(),
            launcher: BackendLauncher::StableDiffusionServer,
            devices: devices.clone(),
            engines: vec![BackendEngine {
                id: id.clone(),
                name: "stable-diffusion.cpp".into(),
                runtime: "stable-diffusion.cpp".into(),
                description: format!("官方 stable-diffusion.cpp {label} 构建"),
                vendor: "stable-diffusion.cpp".into(),
                variant: label.into(),
                category: "image".into(),
                homepage: Some(release.html_url.clone()),
                implementations: vec!["sd.txt2img".into()],
                formats: vec!["safetensors".into(), "gguf".into()],
                modalities: vec!["text".into(), "image".into()],
                devices,
            }],
            files,
        });
    }
    backends
}

fn find_windows_asset<'a>(
    assets: &'a [GithubAsset],
    prefix: &str,
    variant: &str,
) -> Option<&'a GithubAsset> {
    let variant = variant.to_ascii_lowercase();
    assets.iter().find(|asset| {
        let name = asset.name.to_ascii_lowercase();
        name.starts_with(prefix)
            && name.ends_with(".zip")
            && name.contains("bin-win")
            && name.contains(&variant)
            && (prefix == "cudart" || !name.contains("cudart"))
            && !name.contains("arm64")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> GithubAsset {
        GithubAsset {
            name: name.into(),
            browser_download_url: format!("https://example.com/{name}"),
            size: 1,
            digest: None,
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
            find_windows_asset(&assets, "llama", "Vulkan").map(|asset| asset.name.as_str()),
            Some("llama-b1-bin-win-vulkan-x64.zip")
        );
        assert_eq!(
            find_windows_asset(&assets, "llama", "CUDA").map(|asset| asset.name.as_str()),
            Some("llama-b1-bin-win-cuda-12.4-x64.zip")
        );
        assert_eq!(
            find_windows_asset(&assets, "llama", "CPU").map(|asset| asset.name.as_str()),
            Some("llama-b1-bin-win-cpu-x64.zip")
        );
        assert!(find_windows_asset(&assets, "llama", "ROCm").is_none());
    }

    #[test]
    fn selects_stable_diffusion_assets() {
        let assets = vec![
            asset("sd-master-532d2c1-bin-win-vulkan-x64.zip"),
            asset("sd-master-532d2c1-bin-win-cuda12-x64.zip"),
            asset("cudart-sd-bin-win-cu12-x64.zip"),
            asset("sd-master-532d2c1-bin-win-cpu-x64.zip"),
        ];
        assert_eq!(
            find_windows_asset(&assets, "sd-", "Vulkan").map(|asset| asset.name.as_str()),
            Some("sd-master-532d2c1-bin-win-vulkan-x64.zip")
        );
        assert_eq!(
            find_windows_asset(&assets, "sd-", "CPU").map(|asset| asset.name.as_str()),
            Some("sd-master-532d2c1-bin-win-cpu-x64.zip")
        );
        assert_eq!(
            find_windows_asset(&assets, "cudart", "cuda").map(|asset| asset.name.as_str()),
            Some("cudart-sd-bin-win-cu12-x64.zip")
        );
    }
}
