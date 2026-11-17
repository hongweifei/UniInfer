use uniinfer_backend_store::{BackendLauncher, BackendManifest};
use uniinfer_core::DeviceClass;

use super::super::manifests::{CatalogSpec, VariantSpec, catalog_from_variants};
use super::super::{GithubAsset, GithubRelease};

const SPEC: CatalogSpec = CatalogSpec {
    name: "ONNX Runtime",
    id_prefix: "onnxruntime",
    worker: "onnxruntime.dll",
    launcher: BackendLauncher::OnnxRuntime,
    runtime: "onnx",
    category: "language",
    implementations: &[
        "onnx.graph",
        "onnx.pipeline",
        "onnx.diffusion",
        "onnx.layered",
    ],
    formats: &["onnx"],
    modalities: &["text", "audio", "image", "video"],
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
        if !name.starts_with("onnxruntime-win-x64-")
            || !name.ends_with(".zip")
            || name.contains("arm64")
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
    if name.contains("-gpu_cuda12-") {
        return Some(("cuda12".into(), "CUDA 12".into(), vec![DeviceClass::Gpu]));
    }
    if name.contains("-gpu_cuda13-") {
        return Some(("cuda13".into(), "CUDA 13".into(), vec![DeviceClass::Gpu]));
    }
    if name.starts_with("onnxruntime-win-x64-") {
        return Some(("cpu".into(), "CPU".into(), vec![DeviceClass::Cpu]));
    }
    None
}
