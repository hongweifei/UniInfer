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
    match github_releases(&client, "ggml-org/llama.cpp").await {
        Ok((releases, mirrored)) => {
            if let Some(release) = releases.into_iter().find(has_llama_windows_assets) {
                catalog.backends.extend(llama_catalog(&release, mirrored));
            } else {
                errors.push("llama.cpp: 最近的 release 没有 Windows 构建".into());
            }
        }
        Err(error) => errors.push(format!("llama.cpp: {error}")),
    }
    match github_releases(&client, "leejet/stable-diffusion.cpp").await {
        Ok((releases, mirrored)) => {
            if let Some(release) = releases.into_iter().find(has_sd_windows_assets) {
                catalog.backends.extend(sd_catalog(&release, mirrored));
            } else {
                errors.push("stable-diffusion.cpp: 最近的 release 没有 Windows 构建".into());
            }
        }
        Err(error) => errors.push(format!("stable-diffusion.cpp: {error}")),
    }
    if catalog.backends.is_empty() {
        return Err(format!(
            "官方 release 没有可用的 Windows 推理引擎：{}",
            errors.join("；")
        ));
    }
    catalog.warnings = errors;
    for package in &catalog.backends {
        validate_manifest(package).map_err(|error| error.to_string())?;
    }
    Ok(catalog)
}

async fn github_releases(
    client: &reqwest::Client,
    repo: &str,
) -> Result<(Vec<GithubRelease>, bool), String> {
    let direct = format!("https://api.github.com/repos/{repo}/releases?per_page=20");
    let api_error = match fetch_releases(client, &direct).await {
        Ok(releases) => return Ok((releases, false)),
        Err(error) => format!("直连：{error}"),
    };
    let mirrored = format!("{GITHUB_MIRROR}{direct}");
    let api_error = match fetch_releases(client, &mirrored).await {
        Ok(releases) => return Ok((releases, true)),
        Err(error) => format!("{api_error}；镜像：{error}"),
    };
    github_atom_releases(client, repo)
        .await
        .map_err(|atom_error| {
            format!(
                "无法访问 GitHub（{api_error}；Atom：{atom_error}）。请检查网络或代理，也可以稍后重试。"
            )
        })
}

fn has_llama_windows_assets(release: &GithubRelease) -> bool {
    find_windows_asset(&release.assets, "llama", "CPU").is_some()
}

fn has_sd_windows_assets(release: &GithubRelease) -> bool {
    find_windows_asset(&release.assets, "sd-", "CPU").is_some()
}

async fn fetch_releases(client: &reqwest::Client, url: &str) -> Result<Vec<GithubRelease>, String> {
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

async fn github_atom_releases(
    client: &reqwest::Client,
    repo: &str,
) -> Result<(Vec<GithubRelease>, bool), String> {
    let direct = format!("https://github.com/{repo}/releases.atom");
    let (body, mirrored) = match fetch_text(client, &direct).await {
        Ok(body) => (body, false),
        Err(direct_error) => {
            let mirrored = format!("{GITHUB_MIRROR}{direct}");
            let body = fetch_text(client, &mirrored)
                .await
                .map_err(|mirror_error| format!("直连：{direct_error}；镜像：{mirror_error}"))?;
            (body, true)
        }
    };
    let releases = parse_atom_releases(&body, repo);
    if releases.is_empty() {
        return Err("Atom feed 没有可解析的 release".into());
    }
    Ok((releases, mirrored))
}

async fn fetch_text(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let response = client
        .get(url)
        .timeout(Duration::from_secs(90))
        .header(reqwest::header::USER_AGENT, "UniInfer")
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    response.text().await.map_err(|error| error.to_string())
}

fn parse_atom_releases(body: &str, repo: &str) -> Vec<GithubRelease> {
    let mut releases = Vec::new();
    for entry in body.split("<entry>").skip(1) {
        let Some(end) = entry.find("</entry>") else {
            continue;
        };
        let entry = &entry[..end];
        let Some(tag) = between(entry, "<title>", "</title>") else {
            continue;
        };
        let tag = tag.trim();
        if tag.is_empty() {
            continue;
        }
        let prefix = format!("https://github.com/{repo}/releases/download/{tag}/");
        let mut assets = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut rest = entry;
        while let Some(index) = rest.find(&prefix) {
            let candidate = &rest[index..];
            let end = candidate
                .find(|character: char| {
                    character.is_whitespace() || matches!(character, '"' | '<' | '>' | '&')
                })
                .unwrap_or(candidate.len());
            let url = &candidate[..end];
            let name = url.rsplit('/').next().unwrap_or_default();
            if name.ends_with(".zip") && seen.insert(name.to_owned()) {
                assets.push(GithubAsset {
                    name: name.to_owned(),
                    browser_download_url: url.to_owned(),
                    size: 0,
                    digest: None,
                });
            }
            rest = &candidate[end..];
        }
        if !assets.is_empty() {
            releases.push(GithubRelease {
                tag_name: tag.to_owned(),
                html_url: format!("https://github.com/{repo}/releases/tag/{tag}"),
                assets,
            });
        }
    }
    releases
}

fn between<'a>(value: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let from = value.find(start)? + start.len();
    let to = value[from..].find(end)? + from;
    Some(&value[from..to])
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
        ("ROCm", "ROCm", vec![uniinfer_core::DeviceClass::Gpu]),
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
        ("ROCm", "ROCm", vec![uniinfer_core::DeviceClass::Gpu]),
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
            asset("llama-b1-bin-win-rocm-10.0-x64.zip"),
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
        assert_eq!(
            find_windows_asset(&assets, "llama", "ROCm").map(|asset| asset.name.as_str()),
            Some("llama-b1-bin-win-rocm-10.0-x64.zip")
        );
    }

    #[test]
    fn parses_release_assets_from_atom_feed() {
        let atom = r#"<feed><entry><title>b123</title><content type="html">
            &lt;a href="https://github.com/ggml-org/llama.cpp/releases/download/b123/llama-b123-bin-win-cpu-x64.zip"&gt;
            &lt;a href="https://github.com/ggml-org/llama.cpp/releases/download/b123/llama-b123-bin-win-rocm-10.0-x64.zip"&gt;
            &lt;a href="https://github.com/ggml-org/llama.cpp/releases/download/b123/llama-b123-bin-win-vulkan-arm64.zip"&gt;
        </content></entry></feed>"#;
        let releases = parse_atom_releases(atom, "ggml-org/llama.cpp");
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].tag_name, "b123");
        assert_eq!(releases[0].assets.len(), 3);
        assert!(
            releases[0]
                .assets
                .iter()
                .any(|asset| asset.name == "llama-b123-bin-win-rocm-10.0-x64.zip")
        );
    }

    #[test]
    fn builds_one_package_per_supported_variant() {
        let release = GithubRelease {
            tag_name: "b12466".into(),
            html_url: "https://example.com/llama".into(),
            assets: vec![
                asset("llama-b12466-bin-win-vulkan-x64.zip"),
                asset("llama-b12466-bin-win-cuda-12.4-x64.zip"),
                asset("llama-b12466-bin-win-rocm-10.0-x64.zip"),
                asset("llama-b12466-bin-win-cpu-x64.zip"),
                asset("llama-b12466-bin-win-vulkan-arm64.zip"),
            ],
        };
        let packages = llama_catalog(&release, false);
        assert_eq!(packages.len(), 4);
        assert!(
            packages
                .iter()
                .any(|package| package.id == "llama-cpp-rocm")
        );
        assert!(packages.iter().all(|package| package.name == "llama.cpp"));

        let stable = GithubRelease {
            tag_name: "b1".into(),
            html_url: "https://example.com/sd".into(),
            assets: vec![
                asset("sd-b1-bin-win-vulkan-x64.zip"),
                asset("sd-b1-bin-win-cuda12-x64.zip"),
                asset("sd-b1-bin-win-rocm-10.1.0-x64.zip"),
                asset("sd-b1-bin-win-cpu-x64.zip"),
            ],
        };
        let packages = sd_catalog(&stable, false);
        assert_eq!(packages.len(), 4);
        assert!(
            packages
                .iter()
                .any(|package| package.id == "stable-diffusion-rocm")
        );
    }

    #[test]
    fn selects_stable_diffusion_assets() {
        let assets = vec![
            asset("sd-master-532d2c1-bin-win-vulkan-x64.zip"),
            asset("sd-master-532d2c1-bin-win-cuda12-x64.zip"),
            asset("sd-master-532d2c1-bin-win-rocm-10.1.0-x64.zip"),
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
            find_windows_asset(&assets, "sd-", "ROCm").map(|asset| asset.name.as_str()),
            Some("sd-master-532d2c1-bin-win-rocm-10.1.0-x64.zip")
        );
        assert_eq!(
            find_windows_asset(&assets, "cudart", "cuda").map(|asset| asset.name.as_str()),
            Some("cudart-sd-bin-win-cu12-x64.zip")
        );
    }
}
