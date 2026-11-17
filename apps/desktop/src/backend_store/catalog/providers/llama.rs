use uniinfer_backend_store::{BackendLauncher, BackendManifest};

use super::super::GithubRelease;
use super::super::manifests::{CatalogSpec, VariantSpec, catalog_from_variants};
use super::super::variants::{cudart_asset, windows_variants};

const SPEC: CatalogSpec = CatalogSpec {
    name: "llama.cpp",
    id_prefix: "llama-cpp",
    worker: "llama-server.exe",
    launcher: BackendLauncher::LlamaServer,
    runtime: "llama",
    category: "language",
    implementations: &["llama.text"],
    formats: &["gguf"],
    modalities: &["text"],
    license: "MIT",
};

pub(crate) fn has_windows_assets(release: &GithubRelease) -> bool {
    !windows_variants(&release.assets, "llama-").is_empty()
}

pub(crate) fn catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    let variants = windows_variants(&release.assets, "llama-")
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
