use uniinfer_backend_store::{BackendCatalog, validate_manifest};

mod github;
mod manifests;
mod variants;

pub(super) const GITHUB_MIRROR: &str = "https://gh-proxy.com/";

#[derive(serde::Deserialize)]
pub(super) struct GithubRelease {
    pub(super) tag_name: String,
    pub(super) html_url: String,
    pub(super) assets: Vec<GithubAsset>,
}

#[derive(Clone, serde::Deserialize)]
pub(super) struct GithubAsset {
    pub(super) name: String,
    pub(super) browser_download_url: String,
    pub(super) size: u64,
    #[serde(default)]
    pub(super) digest: Option<String>,
}

pub(super) async fn fetch() -> Result<BackendCatalog, String> {
    let client = reqwest::Client::new();
    let mut catalog = BackendCatalog::default();
    let mut errors = Vec::new();
    match github::github_releases(&client, "ggml-org/llama.cpp").await {
        Ok((releases, mirrored)) => {
            if let Some(release) = releases
                .into_iter()
                .find(variants::has_llama_windows_assets)
            {
                catalog
                    .backends
                    .extend(manifests::llama_catalog(&release, mirrored));
            } else {
                errors.push("llama.cpp: 最近的 release 没有 Windows 构建".into());
            }
        }
        Err(error) => errors.push(format!("llama.cpp: {error}")),
    }
    match github::github_releases(&client, "leejet/stable-diffusion.cpp").await {
        Ok((releases, mirrored)) => {
            if let Some(release) = releases.into_iter().find(variants::has_sd_windows_assets) {
                catalog
                    .backends
                    .extend(manifests::sd_catalog(&release, mirrored));
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

#[cfg(test)]
mod tests {
    use super::github::{parse_atom_releases, parse_expanded_assets};
    use super::manifests::{llama_catalog, sd_catalog};
    use super::variants::{cudart_asset, windows_variants};
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
    fn discovers_every_x64_variant_without_an_allowlist() {
        let assets = vec![
            asset("llama-b1-bin-win-vulkan-x64.zip"),
            asset("llama-b1-bin-win-cuda-12.4-x64.zip"),
            asset("llama-b1-bin-win-cuda-13.5-x64.zip"),
            asset("llama-b1-bin-win-rocm-10.0-x64.zip"),
            asset("llama-b1-bin-win-sycl-x64.zip"),
            asset("llama-b1-bin-win-openvino-2026.4.1-x64.zip"),
            asset("cudart-llama-bin-win-cu12.4-x64.zip"),
            asset("llama-b1-bin-win-cpu-x64.zip"),
            asset("llama-b1-bin-win-vulkan-arm64.zip"),
            asset("llama-b1-xcframework.zip"),
        ];
        let variants = windows_variants(&assets, "llama-");
        let labels = variants
            .iter()
            .map(|variant| variant.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "Vulkan",
                "CUDA 12.4",
                "CUDA 13.5",
                "ROCm 10.0",
                "SYCL",
                "OpenVINO 2026.4.1",
                "CPU"
            ]
        );
        assert_eq!(variants[6].devices, vec![uniinfer_core::DeviceClass::Cpu]);
        assert_eq!(variants[0].devices, vec![uniinfer_core::DeviceClass::Gpu]);
    }

    #[test]
    fn pairs_cudart_with_each_cuda_version() {
        let assets = vec![
            asset("cudart-llama-bin-win-cu12.4-x64.zip"),
            asset("cudart-llama-bin-win-cu13.5-x64.zip"),
        ];
        assert_eq!(
            cudart_asset(&assets, "cuda-12.4").map(|asset| asset.name.as_str()),
            Some("cudart-llama-bin-win-cu12.4-x64.zip")
        );
        assert_eq!(
            cudart_asset(&assets, "cuda-13.5").map(|asset| asset.name.as_str()),
            Some("cudart-llama-bin-win-cu13.5-x64.zip")
        );
    }

    #[test]
    fn parses_assets_from_expanded_release_html() {
        let html = r#"<ul><li class="Box-row">
            <a href="/leejet/stable-diffusion.cpp/releases/download/tag/sd-tag-bin-win-cuda12-x64.zip">sd</a>
            <span>sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</span>
        </li><li class="Box-row">
            <a href="/leejet/stable-diffusion.cpp/releases/download/tag/sd-tag-bin-win-vulkan-x64.zip">sd</a>
        </li></ul>"#;
        let assets = parse_expanded_assets(html, "leejet/stable-diffusion.cpp", "tag");
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0].name, "sd-tag-bin-win-cuda12-x64.zip");
        assert_eq!(
            assets[0].digest.as_deref(),
            Some("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        assert_eq!(assets[1].name, "sd-tag-bin-win-vulkan-x64.zip");
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
    fn builds_packages_for_every_variant_and_companion_file() {
        let release = GithubRelease {
            tag_name: "b12466".into(),
            html_url: "https://example.com/llama".into(),
            assets: vec![
                asset("llama-b12466-bin-win-vulkan-x64.zip"),
                asset("llama-b12466-bin-win-cuda-12.4-x64.zip"),
                asset("llama-b12466-bin-win-cuda-13.5-x64.zip"),
                asset("llama-b12466-bin-win-rocm-10.0-x64.zip"),
                asset("llama-b12466-bin-win-sycl-x64.zip"),
                asset("llama-b12466-bin-win-openvino-2026.4.1-x64.zip"),
                asset("llama-b12466-bin-win-cpu-x64.zip"),
                asset("cudart-llama-bin-win-cu12.4-x64.zip"),
                asset("cudart-llama-bin-win-cu13.5-x64.zip"),
                asset("llama-b12466-bin-win-vulkan-arm64.zip"),
            ],
        };
        let packages = llama_catalog(&release, false);
        assert_eq!(packages.len(), 7);
        assert!(packages.iter().all(|package| package.name == "llama.cpp"));
        assert!(
            packages
                .iter()
                .any(|package| package.id == "llama-cpp-rocm-10.0")
        );
        assert!(
            packages
                .iter()
                .any(|package| package.id == "llama-cpp-sycl")
        );
        assert!(
            packages
                .iter()
                .any(|package| package.id == "llama-cpp-openvino-2026.4.1")
        );
        assert!(
            packages
                .iter()
                .find(|package| package.id == "llama-cpp-cuda-12.4")
                .is_some_and(|package| package.files.len() == 2)
        );
        assert!(
            packages
                .iter()
                .find(|package| package.id == "llama-cpp-cuda-13.5")
                .is_some_and(|package| package.files.len() == 2)
        );

        let stable = GithubRelease {
            tag_name: "b1".into(),
            html_url: "https://example.com/sd".into(),
            assets: vec![
                asset("sd-b1-bin-win-vulkan-x64.zip"),
                asset("sd-b1-bin-win-cuda12-x64.zip"),
                asset("sd-b1-bin-win-rocm-10.1.0-x64.zip"),
                asset("sd-b1-bin-win-cpu-x64.zip"),
                asset("cudart-sd-bin-win-cu12-x64.zip"),
            ],
        };
        let packages = sd_catalog(&stable, false);
        assert_eq!(packages.len(), 4);
        assert!(
            packages
                .iter()
                .any(|package| package.id == "stable-diffusion-rocm-10.1.0")
        );
        assert!(
            packages
                .iter()
                .find(|package| package.id == "stable-diffusion-cuda12")
                .is_some_and(|package| package.files.len() == 2)
        );
    }
}
