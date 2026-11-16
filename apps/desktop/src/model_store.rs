//! Online model search, download planning and installation.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;
use uniinfer_core::ModelManifest;
use uniinfer_model_inspect::{InspectOptions, inspect};

mod plan;
mod repo;
mod search;

use plan::{build_plan_with_variant, build_variants, choose_variant};
use repo::fetch_repo_files;
use search::{search_huggingface, search_modelscope};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Source {
    HuggingFace,
    ModelScope,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SearchPage {
    pub(crate) items: Vec<SearchResult>,
    pub(crate) page: u32,
    pub(crate) page_size: u32,
    pub(crate) has_more: bool,
    pub(crate) total: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SearchResult {
    pub(crate) id: String,
    pub(crate) downloads: u64,
    pub(crate) task: String,
    pub(crate) tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PlanFile {
    pub(crate) path: String,
    pub(crate) size: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PlanVariant {
    pub(crate) key: String,
    pub(crate) label: String,
    pub(crate) source: String,
    pub(crate) format: String,
    pub(crate) quantization: String,
    pub(crate) total: u64,
    pub(crate) files: usize,
    pub(crate) recommended: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct InstallPlan {
    pub(crate) repo: String,
    pub(crate) format: String,
    pub(crate) files: Vec<PlanFile>,
    pub(crate) total: u64,
    pub(crate) default_variant: String,
    pub(crate) variants: Vec<PlanVariant>,
}

#[derive(Clone, Debug)]
struct RemoteFile {
    repo: String,
    path: String,
    url: String,
    sha256: Option<String>,
    size: u64,
}

#[derive(Clone, Serialize)]
struct DownloadEvent {
    repo: String,
    file: String,
    downloaded: u64,
    total: u64,
    status: &'static str,
    message: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct InstallResult {
    pub(crate) id: String,
    pub(crate) path: String,
    pub(crate) restart_required: bool,
}

#[tauri::command]
pub(crate) async fn search_models(
    query: String,
    source: Source,
    page: Option<u32>,
    page_size: Option<u32>,
) -> Result<SearchPage, String> {
    let query = query.trim();
    let page = page.unwrap_or(1).max(1);
    let page_size = page_size.unwrap_or(10).clamp(1, 40);
    if query.is_empty() {
        return Ok(SearchPage {
            items: Vec::new(),
            page,
            page_size,
            has_more: false,
            total: None,
        });
    }
    match source {
        Source::HuggingFace => search_huggingface(query, page, page_size).await,
        Source::ModelScope => search_modelscope(query, page, page_size).await,
    }
}

#[tauri::command]
pub(crate) async fn plan_model(repo: String, source: Source) -> Result<InstallPlan, String> {
    let (repo_files, base_files) = fetch_repo_files(&repo, source).await?;
    let variants = build_variants(&repo_files, base_files.as_ref())?;
    let chosen = choose_variant(&variants, None)?;
    Ok(InstallPlan {
        repo,
        format: chosen.format.clone(),
        default_variant: chosen.key.clone(),
        total: chosen.files.iter().map(|file| file.size).sum(),
        files: chosen
            .files
            .iter()
            .map(|file| PlanFile {
                path: file.path.clone(),
                size: file.size,
            })
            .collect(),
        variants: variants
            .iter()
            .map(|variant| PlanVariant {
                key: variant.key.clone(),
                label: variant.label.clone(),
                source: variant.source.clone(),
                format: variant.format.clone(),
                quantization: variant.quantization.clone(),
                total: variant.files.iter().map(|file| file.size).sum(),
                files: variant.files.len(),
                recommended: variant.recommended,
            })
            .collect(),
    })
}

#[derive(Serialize)]
pub(crate) struct DeleteResult {
    pub(crate) id: String,
    pub(crate) removed: bool,
    pub(crate) restart_required: bool,
}

#[tauri::command]
pub(crate) async fn delete_model(id: String) -> Result<DeleteResult, String> {
    tokio::task::spawn_blocking(move || delete_model_inner(&id))
        .await
        .map_err(|error| format!("delete task: {error}"))?
}

fn delete_model_inner(id: &str) -> Result<DeleteResult, String> {
    let base = models_root()
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let targets = crate::api_server::manifest_paths()
        .into_iter()
        .filter(|path| {
            ModelManifest::load(path)
                .map(|manifest| manifest.id == id)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if targets.is_empty() {
        return Err(format!("找不到模型 {id} 的 manifest"));
    }
    let models_root = base.join("models");
    let mut removed = false;
    for manifest in targets {
        let Some(dir) = manifest.parent() else {
            continue;
        };
        if !dir.starts_with(&models_root) || dir == models_root {
            return Err(format!("拒绝删除模型目录外的路径：{}", dir.display()));
        }
        if dir.exists() {
            std::fs::remove_dir_all(dir)
                .map_err(|error| format!("删除 {}: {error}", dir.display()))?;
            removed = true;
        }
    }

    Ok(DeleteResult {
        id: id.to_string(),
        removed,
        restart_required: false,
    })
}

#[tauri::command]
pub(crate) async fn install_model(
    app: AppHandle,
    repo: String,
    source: Source,
    variant: Option<String>,
) -> Result<InstallResult, String> {
    let (repo_files, base_files) = fetch_repo_files(&repo, source).await?;
    let plan = build_plan_with_variant(&repo_files, base_files.as_ref(), variant.as_deref())?;
    let id = slug(&repo);
    let target = models_root().join(&id);
    let staging = temp_model_path(&models_root(), &id, "installing");
    if staging.exists() {
        tokio::fs::remove_dir_all(&staging)
            .await
            .map_err(|error| format!("清理临时目录 {}: {error}", staging.display()))?;
    }
    tokio::fs::create_dir_all(&staging)
        .await
        .map_err(|error| format!("create {}: {error}", staging.display()))?;
    let _staging_cleanup = StagingCleanup(staging.clone());
    let total = plan.files.iter().map(|file| file.size).sum::<u64>();
    if source == Source::ModelScope {
        let files = plan.files.clone();
        let staging_dir = staging.clone();
        let app_handle = app.clone();
        tokio::task::spawn_blocking(move || download_modelscope(&app_handle, &files, &staging_dir))
            .await
            .map_err(|error| format!("modelscope download task: {error}"))??;
    } else {
        let client = reqwest::Client::builder()
            .user_agent("UniInfer-Desktop/0.0")
            .build()
            .map_err(|error| format!("http client: {error}"))?;
        let mut done = 0u64;
        for file in &plan.files {
            let target = staging.join(&file.path);
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|error| format!("create {}: {error}", parent.display()))?;
            }
            if target.is_file() {
                done += file.size;
                continue;
            }
            let response = client
                .get(&file.url)
                .send()
                .await
                .map_err(|error| format!("download {}: {error}", file.url))?;
            if !response.status().is_success() {
                return Err(format!("download {}: HTTP {}", file.url, response.status()));
            }
            let part = target.with_extension("part");
            let mut output = tokio::fs::File::create(&part)
                .await
                .map_err(|error| format!("create {}: {error}", part.display()))?;
            let mut hasher = Sha256::new();
            let mut current = 0u64;
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|error| format!("download {}: {error}", file.url))?;
                current += chunk.len() as u64;
                hasher.update(&chunk);
                output
                    .write_all(&chunk)
                    .await
                    .map_err(|error| format!("write {}: {error}", part.display()))?;
                let _ = app.emit(
                    "model-download",
                    DownloadEvent {
                        repo: repo.clone(),
                        file: file.path.clone(),
                        downloaded: done + current,
                        total: total.max(1),
                        status: "downloading",
                        message: None,
                    },
                );
            }
            output
                .flush()
                .await
                .map_err(|error| format!("flush {}: {error}", part.display()))?;
            drop(output);
            if let Some(expected) = &file.sha256 {
                let actual = hex::encode(hasher.finalize());
                if !actual.eq_ignore_ascii_case(expected) {
                    let _ = tokio::fs::remove_file(&part).await;
                    return Err(format!(
                        "checksum mismatch for {}: expected {expected}, got {actual}",
                        file.path
                    ));
                }
            }
            tokio::fs::rename(&part, &target)
                .await
                .map_err(|error| format!("rename {}: {error}", target.display()))?;
            done += file.size;
        }
    }
    let directory = staging.clone();
    let model_id = id.clone();
    let _manifest = tokio::task::spawn_blocking(move || -> Result<PathBuf, String> {
        let inspection = inspect(
            &directory,
            &InspectOptions {
                id: Some(model_id),
                ..InspectOptions::default()
            },
        )
        .map_err(|error| format!("无法为下载的模型生成 manifest: {error}"))?;
        inspection
            .write_manifest()
            .map_err(|error| format!("写入 manifest: {error}"))
    })
    .await
    .map_err(|error| format!("inspect task: {error}"))??;
    let _ = app.emit(
        "model-download",
        DownloadEvent {
            repo: repo.clone(),
            file: String::new(),
            downloaded: total,
            total: total.max(1),
            status: "done",
            message: Some("下载完成，默认运行配置已生成".into()),
        },
    );
    let commit_staging = staging.clone();
    let commit_target = target.clone();
    tokio::task::spawn_blocking(move || commit_install(&commit_staging, &commit_target))
        .await
        .map_err(|error| format!("install commit task: {error}"))??;
    Ok(InstallResult {
        id,
        path: target.display().to_string(),
        restart_required: false,
    })
}

fn slug(repo: &str) -> String {
    repo.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

struct StagingCleanup(PathBuf);

impl Drop for StagingCleanup {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn models_root() -> PathBuf {
    crate::api_server::models_root()
}

fn temp_model_path(parent: &Path, id: &str, kind: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    parent.join(format!(".{id}.{kind}-{nanos}"))
}

fn commit_install(staging: &Path, target: &Path) -> Result<(), String> {
    let backup = temp_model_path(
        target.parent().unwrap_or_else(|| Path::new(".")),
        target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model"),
        "backup",
    );
    let had_target = target.exists();
    if had_target {
        std::fs::rename(target, &backup)
            .map_err(|error| format!("暂存旧模型 {}: {error}", target.display()))?;
    }
    if let Err(error) = std::fs::rename(staging, target) {
        if had_target {
            let _ = std::fs::rename(&backup, target);
        }
        return Err(format!("提交模型 {}: {error}", target.display()));
    }
    if had_target {
        let _ = std::fs::remove_dir_all(&backup);
    }
    Ok(())
}

#[derive(Clone)]
struct CatProgress {
    app: AppHandle,
    repo: String,
    done: u64,
    total: u64,
}

impl models_cat::Progress for CatProgress {
    fn on_start(&mut self, unit: &models_cat::ProgressUnit) -> Result<(), models_cat::OpsError> {
        let _ = self.app.emit(
            "model-download",
            DownloadEvent {
                repo: self.repo.clone(),
                file: unit.filename().to_string(),
                downloaded: self.done,
                total: self.total.max(1),
                status: "downloading",
                message: None,
            },
        );
        Ok(())
    }

    fn on_progress(&mut self, unit: &models_cat::ProgressUnit) -> Result<(), models_cat::OpsError> {
        let _ = self.app.emit(
            "model-download",
            DownloadEvent {
                repo: self.repo.clone(),
                file: unit.filename().to_string(),
                downloaded: self.done + unit.current(),
                total: self.total.max(1),
                status: "downloading",
                message: None,
            },
        );
        Ok(())
    }

    fn on_finish(&mut self, _unit: &models_cat::ProgressUnit) -> Result<(), models_cat::OpsError> {
        Ok(())
    }
}

fn download_modelscope(app: &AppHandle, files: &[RemoteFile], target: &Path) -> Result<(), String> {
    let cache = models_root().join(".modelscope-cache");
    let mut done = 0u64;
    let total = files.iter().map(|file| file.size).sum::<u64>();
    for file in files {
        let dest = target.join(&file.path);
        if dest.is_file() {
            done += file.size;
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let mut repo = models_cat::Repo::new_model(&file.repo);
        repo.set_cache_dir(&cache);
        let cat = models_cat::ModelsCat::new(repo.clone());
        let progress = CatProgress {
            app: app.clone(),
            repo: file.repo.clone(),
            done,
            total,
        };
        cat.download_with_progress(&file.path, progress)
            .map_err(|error| format!("modelscope download {}: {error}", file.path))?;
        let source = locate_cache_file(&repo, &file.path)?;
        std::fs::copy(&source, &dest)
            .map_err(|error| format!("copy {}: {error}", dest.display()))?;
        done += file.size;
        let _ = std::fs::remove_dir_all(repo.cache_dir());
    }
    Ok(())
}

fn locate_cache_file(repo: &models_cat::Repo, path: &str) -> Result<PathBuf, String> {
    let snapshots = repo.cache_dir().join("snapshots");
    let entries = std::fs::read_dir(&snapshots)
        .map_err(|error| format!("read {}: {error}", snapshots.display()))?;
    for entry in entries.flatten() {
        let candidate = entry.path().join(path);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!("downloaded file not found in cache: {path}"))
}

#[cfg(test)]
mod tests {
    use super::plan::{build_plan, build_plan_with_variant, build_variants};
    use super::repo::{RemoteInfo, RepoFiles, fetch_hf_repo, fetch_modelscope_repo};
    use super::*;

    fn repo(repo: &str, source: Source, files: &[(&str, u64)]) -> RepoFiles {
        RepoFiles {
            repo: repo.into(),
            source,
            files: files
                .iter()
                .map(|(path, size)| RemoteInfo {
                    path: (*path).into(),
                    size: *size,
                    sha256: None,
                })
                .collect(),
            base_model: None,
        }
    }

    #[test]
    fn gguf_plan_prefers_q4_k_m_and_base_tokenizer() {
        let model = repo(
            "unsloth/Qwen3-0.6B-GGUF",
            Source::HuggingFace,
            &[
                ("Qwen3-0.6B-Q8_0.gguf", 10),
                ("Qwen3-0.6B-Q4_K_M.gguf", 20),
                ("config.json", 1),
            ],
        );
        let base = repo(
            "Qwen/Qwen3-0.6B",
            Source::HuggingFace,
            &[
                ("tokenizer.json", 1),
                ("tokenizer_config.json", 1),
                ("model.safetensors", 100),
            ],
        );
        let plan = build_plan(&model, Some(&base)).expect("plan");
        assert_eq!(plan.format, "gguf");
        assert!(
            plan.files
                .iter()
                .any(|file| file.path.ends_with("Q4_K_M.gguf"))
        );
        assert!(plan.files.iter().any(|file| file.path == "tokenizer.json"));
        assert!(
            !plan
                .files
                .iter()
                .any(|file| file.path.ends_with("Q8_0.gguf"))
        );
        assert!(
            plan.files
                .iter()
                .all(|file| file.url.contains("hf-mirror.com"))
        );
    }

    #[test]
    fn gguf_variants_are_exposed_and_selectable() {
        let model = repo(
            "unsloth/Qwen3-0.6B-GGUF",
            Source::HuggingFace,
            &[
                ("Qwen3-0.6B-Q8_0.gguf", 10),
                ("Qwen3-0.6B-Q4_K_M.gguf", 20),
                ("config.json", 1),
            ],
        );
        let base = repo(
            "Qwen/Qwen3-0.6B",
            Source::HuggingFace,
            &[("tokenizer.json", 1)],
        );
        let variants = build_variants(&model, Some(&base)).expect("variants");
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].quantization, "Q4_K_M");
        assert!(variants[0].recommended);
        let selected =
            build_plan_with_variant(&model, Some(&base), Some("gguf:Qwen3-0.6B-Q8_0.gguf"))
                .expect("plan");
        assert!(
            selected
                .files
                .iter()
                .any(|file| file.path.ends_with("Q8_0.gguf"))
        );
        assert!(
            !selected
                .files
                .iter()
                .any(|file| file.path.ends_with("Q4_K_M.gguf"))
        );
    }

    #[test]
    fn onnx_plan_includes_external_data_per_quantization() {
        let model = repo(
            "onnx-community/Qwen3.5-0.8B-ONNX",
            Source::HuggingFace,
            &[
                ("onnx/decoder_model_merged.onnx", 10),
                ("onnx/decoder_model_merged.onnx_data", 100),
                ("onnx/decoder_model_merged_q4.onnx", 5),
                ("onnx/decoder_model_merged_q4.onnx_data", 50),
                ("onnx/embed_tokens.onnx", 2),
                ("onnx/embed_tokens.onnx_data", 20),
                ("onnx/embed_tokens_q4.onnx", 1),
                ("onnx/embed_tokens_q4.onnx_data", 10),
                ("config.json", 1),
            ],
        );
        let variants = build_variants(&model, None).expect("variants");
        assert_eq!(variants.len(), 2);
        for variant in &variants {
            let has_data = variant
                .files
                .iter()
                .any(|file| file.path.ends_with(".onnx_data"));
            assert!(has_data, "variant {} lost external data", variant.key);
        }
        let q4 = variants
            .iter()
            .find(|variant| variant.key == "onnx:onnx:q4")
            .expect("q4 variant");
        assert!(q4.files.iter().any(|file| file.path.ends_with("_q4.onnx")));
        assert!(
            !q4.files
                .iter()
                .any(|file| file.path.ends_with("merged.onnx"))
        );
        assert!(
            q4.files
                .iter()
                .any(|file| file.path.ends_with("_q4.onnx_data"))
        );
    }

    #[test]
    fn safetensors_plan_keeps_shards_and_sidecars() {
        let model = repo(
            "Qwen/Qwen3-0.6B",
            Source::ModelScope,
            &[
                ("model-00001-of-00002.safetensors", 10),
                ("model-00002-of-00002.safetensors", 11),
                ("config.json", 1),
                ("tokenizer.json", 1),
                ("README.md", 4),
            ],
        );
        let plan = build_plan(&model, None).expect("plan");
        assert_eq!(plan.format, "safetensors");
        assert_eq!(
            plan.files
                .iter()
                .filter(|file| file.path.ends_with(".safetensors"))
                .count(),
            2
        );
        assert!(
            !plan
                .files
                .iter()
                .any(|file| file.path.ends_with("README.md"))
        );
        assert!(
            plan.files
                .iter()
                .all(|file| file.url.starts_with("https://modelscope.cn/"))
        );
    }

    #[test]
    #[ignore = "network: probes live Hugging Face ONNX external data"]
    fn live_hf_onnx_plan_keeps_external_data() {
        tauri::async_runtime::block_on(async {
            let repo = fetch_hf_repo("onnx-community/Qwen3.5-0.8B-ONNX")
                .await
                .expect("repo files");
            let variants = build_variants(&repo, None).expect("variants");
            assert!(variants.iter().any(|variant| variant.format == "onnx"));
            for variant in variants.iter().filter(|variant| variant.format == "onnx") {
                assert!(
                    variant
                        .files
                        .iter()
                        .any(|file| file.path.ends_with(".onnx_data")),
                    "variant {} is missing external data",
                    variant.key
                );
            }
        });
    }

    #[test]
    #[ignore = "network: probes live ModelScope recursive file listing"]
    fn live_modelscope_lists_nested_model_files() {
        tauri::async_runtime::block_on(async {
            let repo = fetch_modelscope_repo("HuggingFaceTB/SmolVLM-256M-Instruct")
                .await
                .expect("repo files");
            assert!(
                repo.files
                    .iter()
                    .any(|file| file.path.ends_with(".onnx") && file.path.contains('/'))
            );
            let variants = build_variants(&repo, None).expect("variants");
            assert!(variants.iter().any(|variant| variant.format == "onnx"));
        });
    }

    #[test]
    #[ignore = "network: probes live ModelScope OpenAPI search"]
    fn live_modelscope_search() {
        tauri::async_runtime::block_on(async {
            let page = search_modelscope("qwen3", 1, 3).await.expect("search");
            assert_eq!(page.page, 1);
            assert!(page.total.unwrap_or(0) > 0);
            assert!(!page.items.is_empty());
        });
    }

    #[test]
    fn commit_install_replaces_existing_model_atomically() {
        let dir = tempfile::tempdir().expect("tempdir");
        let staging = dir.path().join("staging");
        let target = dir.path().join("model");
        std::fs::create_dir_all(&staging).expect("staging");
        std::fs::create_dir_all(&target).expect("target");
        std::fs::write(staging.join("new.txt"), b"new").expect("new");
        std::fs::write(target.join("old.txt"), b"old").expect("old");
        commit_install(&staging, &target).expect("commit");
        assert!(target.join("new.txt").is_file());
        assert!(!target.join("old.txt").exists());
        assert!(!staging.exists());
        assert!(std::fs::read_dir(dir.path()).expect("dir").all(|entry| {
            !entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .contains(".backup-")
        }));
    }

    #[test]
    fn slug_is_filesystem_safe() {
        assert_eq!(slug("Qwen/Qwen3-0.6B"), "Qwen-Qwen3-0.6B");
    }
}
