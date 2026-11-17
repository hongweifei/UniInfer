use uniinfer_backend_store::{BackendEngine, BackendFile, BackendLauncher, BackendManifest};

use super::variants::{cudart_asset, variant_slug, windows_variants};
use super::{GITHUB_MIRROR, GithubAsset, GithubRelease};

struct CatalogSpec {
    name: &'static str,
    id_prefix: &'static str,
    asset_prefix: &'static str,
    worker: &'static str,
    launcher: BackendLauncher,
    runtime: &'static str,
    category: &'static str,
    implementations: &'static [&'static str],
    formats: &'static [&'static str],
    modalities: &'static [&'static str],
}

const LLAMA: CatalogSpec = CatalogSpec {
    name: "llama.cpp",
    id_prefix: "llama-cpp",
    asset_prefix: "llama-",
    worker: "llama-server.exe",
    launcher: BackendLauncher::LlamaServer,
    runtime: "llama",
    category: "language",
    implementations: &["llama.text"],
    formats: &["gguf"],
    modalities: &["text"],
};

const STABLE_DIFFUSION: CatalogSpec = CatalogSpec {
    name: "stable-diffusion.cpp",
    id_prefix: "stable-diffusion",
    asset_prefix: "sd-",
    worker: "sd-server.exe",
    launcher: BackendLauncher::StableDiffusionServer,
    runtime: "stable-diffusion.cpp",
    category: "image",
    implementations: &["sd.txt2img"],
    formats: &["safetensors", "gguf"],
    modalities: &["text", "image"],
};

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

fn catalog_from_variants(
    release: &GithubRelease,
    mirrored: bool,
    spec: &CatalogSpec,
) -> Vec<BackendManifest> {
    windows_variants(&release.assets, spec.asset_prefix)
        .into_iter()
        .map(|variant| {
            let id = format!("{}-{}", spec.id_prefix, variant_slug(&variant.key));
            let mut files = vec![asset_file(variant.asset, mirrored)];
            if let Some(cudart) = cudart_asset(&release.assets, &variant.key) {
                files.push(asset_file(cudart, mirrored));
            }
            let description = format!("官方 {} Windows {} 构建", spec.name, variant.label);
            BackendManifest {
                id: id.clone(),
                name: spec.name.into(),
                version: release.tag_name.clone(),
                description: description.clone(),
                homepage: Some(release.html_url.clone()),
                license: Some("MIT".into()),
                platform: "windows-x86_64".into(),
                worker: spec.worker.into(),
                launcher: spec.launcher,
                devices: variant.devices.clone(),
                engines: vec![BackendEngine {
                    id: id.clone(),
                    name: spec.name.into(),
                    runtime: spec.runtime.into(),
                    description,
                    vendor: spec.name.into(),
                    variant: variant.label,
                    category: spec.category.into(),
                    homepage: Some(release.html_url.clone()),
                    implementations: spec
                        .implementations
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect(),
                    formats: spec
                        .formats
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect(),
                    modalities: spec
                        .modalities
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect(),
                    devices: variant.devices,
                }],
                files,
            }
        })
        .collect()
}

pub(super) fn llama_catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    catalog_from_variants(release, mirrored, &LLAMA)
}

pub(super) fn sd_catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    catalog_from_variants(release, mirrored, &STABLE_DIFFUSION)
}
