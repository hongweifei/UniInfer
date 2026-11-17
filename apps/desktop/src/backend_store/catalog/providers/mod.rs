//! Release-specific catalog providers.
//!
//! Each provider only knows how to select its Windows variants and describe
//! its engine. Manifest construction, download metadata and validation are
//! shared by the catalog.

use uniinfer_backend_store::BackendManifest;

use super::GithubRelease;

pub(super) mod llama;
pub(super) mod onnx_runtime;
pub(super) mod stable_diffusion;
pub(super) mod whisper;

pub(super) struct Provider {
    pub(super) name: &'static str,
    pub(super) repo: &'static str,
    pub(super) has_assets: fn(&GithubRelease) -> bool,
    pub(super) build: fn(&GithubRelease, bool) -> Vec<BackendManifest>,
}

pub(super) const PROVIDERS: &[Provider] = &[
    Provider {
        name: "llama.cpp",
        repo: "ggml-org/llama.cpp",
        has_assets: llama::has_windows_assets,
        build: llama::catalog,
    },
    Provider {
        name: "stable-diffusion.cpp",
        repo: "leejet/stable-diffusion.cpp",
        has_assets: stable_diffusion::has_windows_assets,
        build: stable_diffusion::catalog,
    },
    Provider {
        name: "whisper.cpp",
        repo: "ggml-org/whisper.cpp",
        has_assets: whisper::has_windows_assets,
        build: whisper::catalog,
    },
    Provider {
        name: "ONNX Runtime",
        repo: "microsoft/onnxruntime",
        has_assets: onnx_runtime::has_windows_assets,
        build: onnx_runtime::catalog,
    },
];
