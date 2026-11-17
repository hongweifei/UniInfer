use uniinfer_backend_store::{BackendEngine, BackendFile, BackendLauncher, BackendManifest};

use super::variants::variant_slug;
use super::{GITHUB_MIRROR, GithubAsset, GithubRelease};

/// A catalog entry shared by every downloadable backend provider.
pub(super) struct CatalogSpec {
    pub(super) name: &'static str,
    pub(super) id_prefix: &'static str,
    pub(super) worker: &'static str,
    pub(super) launcher: BackendLauncher,
    pub(super) runtime: &'static str,
    pub(super) category: &'static str,
    pub(super) implementations: &'static [&'static str],
    pub(super) formats: &'static [&'static str],
    pub(super) modalities: &'static [&'static str],
    pub(super) license: &'static str,
}

/// One downloadable Windows x64 variant of a provider release.
pub(super) struct VariantSpec<'a> {
    pub(super) key: String,
    pub(super) label: String,
    pub(super) files: Vec<&'a GithubAsset>,
    pub(super) devices: Vec<uniinfer_core::DeviceClass>,
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

pub(super) fn catalog_from_variants(
    release: &GithubRelease,
    mirrored: bool,
    spec: &CatalogSpec,
    variants: Vec<VariantSpec<'_>>,
) -> Vec<BackendManifest> {
    variants
        .into_iter()
        .map(|variant| {
            let id = format!("{}-{}", spec.id_prefix, variant_slug(&variant.key));
            let files = variant
                .files
                .iter()
                .map(|asset| asset_file(asset, mirrored))
                .collect::<Vec<_>>();
            let description = format!("官方 {} Windows {} 构建", spec.name, variant.label);
            BackendManifest {
                id: id.clone(),
                name: spec.name.into(),
                version: release.tag_name.clone(),
                description: description.clone(),
                homepage: Some(release.html_url.clone()),
                license: Some(spec.license.into()),
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
