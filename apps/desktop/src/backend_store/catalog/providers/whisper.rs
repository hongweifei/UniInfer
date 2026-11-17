use uniinfer_backend_store::{BackendLauncher, BackendManifest};
use uniinfer_core::DeviceClass;

use super::super::manifests::{CatalogSpec, VariantSpec, catalog_from_variants};
use super::super::{GithubAsset, GithubRelease};

const SPEC: CatalogSpec = CatalogSpec {
    name: "whisper.cpp",
    id_prefix: "whisper-cpp",
    worker: "whisper-server.exe",
    launcher: BackendLauncher::WhisperServer,
    runtime: "whisper",
    category: "audio",
    implementations: &["whisper.ggml"],
    formats: &["ggml"],
    modalities: &["audio"],
    license: "MIT",
};

pub(crate) fn has_windows_assets(release: &GithubRelease) -> bool {
    !variants(&release.assets).is_empty()
}

pub(crate) fn catalog(release: &GithubRelease, mirrored: bool) -> Vec<BackendManifest> {
    catalog_from_variants(release, mirrored, &SPEC, variants(&release.assets))
}

fn variants(assets: &[GithubAsset]) -> Vec<VariantSpec<'_>> {
    let mut variants = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for asset in assets {
        let name = asset.name.to_ascii_lowercase();
        if !name.starts_with("whisper-")
            || !name.ends_with("-x64.zip")
            || name.contains("arm64")
            || name.contains("win32")
            || name.contains("opencl")
        {
            continue;
        }
        let Some((key, label, devices)) = variant_identity(&name) else {
            continue;
        };
        if !seen.insert(key.clone()) {
            continue;
        }
        variants.push(VariantSpec {
            key,
            label,
            files: vec![asset],
            devices,
        });
    }
    variants
}

fn variant_identity(name: &str) -> Option<(String, String, Vec<DeviceClass>)> {
    if let Some(version) = name
        .strip_prefix("whisper-cublas-")
        .and_then(|rest| rest.strip_suffix("-bin-x64.zip"))
    {
        return Some(cuda_variant(version));
    }
    if let Some(version) = name
        .strip_prefix("whisper-bin-win-cuda-")
        .and_then(|rest| rest.strip_suffix("-x64.zip"))
    {
        return Some(cuda_variant(version));
    }
    if name == "whisper-bin-x64.zip" || name == "whisper-bin-win-cpu-x64.zip" {
        return Some(("cpu".into(), "CPU".into(), vec![DeviceClass::Cpu]));
    }
    if name == "whisper-blas-bin-x64.zip" {
        return Some(("blas".into(), "CPU BLAS".into(), vec![DeviceClass::Cpu]));
    }
    None
}

fn cuda_variant(version: &str) -> (String, String, Vec<DeviceClass>) {
    let version = version.trim_end_matches(".0");
    (
        format!("cuda-{version}"),
        format!("CUDA {version}"),
        vec![DeviceClass::Gpu],
    )
}
