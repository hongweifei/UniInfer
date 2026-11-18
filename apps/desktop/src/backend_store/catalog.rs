use uniinfer_backend_store::{BackendCatalog, validate_manifest};

mod github;
mod manifests;
mod providers;
mod variants;

pub(super) const GITHUB_MIRROR: &str = "https://gh-proxy.com/";

/// Optional catalog for project-built EP worker packages. The asset is
/// published alongside a UniInfer release; absent assets are ignored.
const EXTRA_CATALOG_URL: &str =
    "https://github.com/hongweifei/UniInfer/releases/latest/download/ep-catalog.json";

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
    for provider in providers::PROVIDERS {
        match github::github_releases(&client, provider.repo).await {
            Ok((releases, mirrored)) => {
                if let Some(release) = releases.into_iter().find(provider.has_assets) {
                    catalog
                        .backends
                        .extend((provider.build)(&release, mirrored));
                } else {
                    errors.push(format!(
                        "{}: 最近的 release 没有 Windows 构建",
                        provider.name
                    ));
                }
            }
            Err(error) => errors.push(format!("{}: {error}", provider.name)),
        }
    }
    match extra_catalog(&client).await {
        Ok(extra) => catalog.backends.extend(extra),
        Err(error) => errors.push(format!("EP worker catalog: {error}")),
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

async fn extra_catalog(
    client: &reqwest::Client,
) -> Result<Vec<uniinfer_backend_store::BackendManifest>, String> {
    let mut last_error = String::new();
    for url in [
        EXTRA_CATALOG_URL.to_owned(),
        format!("{GITHUB_MIRROR}{EXTRA_CATALOG_URL}"),
    ] {
        let response = match client.get(&url).send().await {
            Ok(response) => response,
            Err(error) => {
                last_error = error.to_string();
                continue;
            }
        };
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }
        if !response.status().is_success() {
            last_error = format!("HTTP {}", response.status());
            continue;
        }
        let catalog = response
            .json::<BackendCatalog>()
            .await
            .map_err(|error| format!("解析 EP worker catalog: {error}"))?;
        for package in &catalog.backends {
            validate_manifest(package).map_err(|error| error.to_string())?;
        }
        return Ok(catalog.backends);
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::github::{parse_atom_releases, parse_expanded_assets};
    use super::providers::{llama, stable_diffusion};
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
    #[ignore = "network: probes live official backend releases"]
    fn live_catalog_includes_whisper_and_onnx_runtime() {
        tauri::async_runtime::block_on(async {
            let catalog = fetch().await.expect("catalog");
            assert!(catalog.backends.iter().any(|package| {
                package.name == "whisper.cpp"
                    && package.launcher == uniinfer_backend_store::BackendLauncher::WhisperServer
            }));
            assert!(catalog.backends.iter().any(|package| {
                package.name == "ONNX Runtime"
                    && package.launcher == uniinfer_backend_store::BackendLauncher::OnnxRuntime
            }));
        });
    }

    #[test]
    fn whisper_catalog_discovers_cpu_blas_and_cuda() {
        let release = GithubRelease {
            tag_name: "b5454".into(),
            html_url: "https://example.com/whisper".into(),
            assets: vec![
                asset("whisper-bin-x64.zip"),
                asset("whisper-blas-bin-x64.zip"),
                asset("whisper-cublas-11.8.0-bin-x64.zip"),
                asset("whisper-bin-win-cuda-12.4.0-x64.zip"),
                asset("whisper-bin-win-cpu-arm64.zip"),
                asset("whisper-bin-Win32.zip"),
            ],
        };
        let packages = providers::whisper::catalog(&release, true);
        assert_eq!(packages.len(), 4);
        assert_eq!(packages[0].worker, "whisper-server.exe");
        assert_eq!(
            packages[0].launcher,
            uniinfer_backend_store::BackendLauncher::WhisperServer
        );
        assert!(
            packages
                .iter()
                .any(|package| package.id == "whisper-cpp-cuda-12.4")
        );
        assert!(
            packages
                .iter()
                .find(|package| package.id == "whisper-cpp-blas")
                .is_some_and(|package| {
                    package.devices == vec![uniinfer_core::DeviceClass::Cpu]
                })
        );
        assert!(packages.iter().all(|package| {
            package
                .files
                .iter()
                .all(|file| file.url.starts_with(super::GITHUB_MIRROR))
        }));
    }

    #[test]
    fn onnxruntime_catalog_discovers_cpu_and_cuda() {
        let release = GithubRelease {
            tag_name: "v1.32.0".into(),
            html_url: "https://example.com/ort".into(),
            assets: vec![
                asset("onnxruntime-win-x64-1.32.0.zip"),
                asset("onnxruntime-win-x64-gpu_cuda12-1.32.0.zip"),
                asset("onnxruntime-win-x64-gpu_cuda13-1.32.0.zip"),
                asset("onnxruntime-win-arm64-1.32.0.zip"),
                asset("onnxruntime-linux-x64-1.32.0.tgz"),
            ],
        };
        let packages = providers::onnx_runtime::catalog(&release, false);
        assert_eq!(packages.len(), 3);
        assert!(packages.iter().all(|package| {
            package.worker == "onnxruntime.dll"
                && package.launcher == uniinfer_backend_store::BackendLauncher::OnnxRuntime
        }));
        assert!(
            packages
                .iter()
                .any(|package| package.id == "onnxruntime-cuda12")
        );
        assert!(
            packages
                .iter()
                .find(|package| package.id == "onnxruntime-cpu")
                .is_some_and(|package| package.devices == vec![uniinfer_core::DeviceClass::Cpu])
        );
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
        let packages = llama::catalog(&release, false);
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
        let packages = stable_diffusion::catalog(&stable, false);
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
