use uniinfer_backend_store::{BackendLauncher, BackendManifest};

use super::super::GithubRelease;
use super::super::manifests::{CatalogSpec, VariantSpec, catalog_from_variants};
use super::super::variants::{cudart_asset, windows_variants};

const SPEC: CatalogSpec = CatalogSpec {
    name: "stable-diffusion.cpp",
    id_prefix: "stable-diffusion",
    worker: "sd-server.exe",
    launcher: BackendLauncher::StableDiffusionServer,
    runtime: "stable-diffusion.cpp",
    category: "image",
    implementations: &["sd.txt2img"],
    formats: &["safetensors", "gguf"],
    modalities: &["text", "image"],
    license: "MIT",
};

pub(crate) fn has_windows_assets(release: &GithubRelease) -> bool {
    !windows_variants(&release.assets, "sd-").is_empty()
}

pub(crate) fn catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    let variants = windows_variants(&release.assets, "sd-")
        .into_iter()
        .map(|variant| {
            let mut files = vec![variant.asset];
            if let Some(cudart) = cudart_asset(&release.assets, &variant.key) {
                files.push(cudart);
            }
            VariantSpec {
                key: variant.key,
                label: variant.label,
                files,
                devices: variant.devices,
            }
        })
        .collect();
    catalog_from_variants(release, mirrored, &SPEC, variants)
}
