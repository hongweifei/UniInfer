//! Online model search, download planning and installation.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;
use uniinfer_core::ModelManifest;
use uniinfer_model_inspect::{InspectOptions, inspect};

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

#[derive(Clone, Debug)]
struct RepoFiles {
    repo: String,
    source: Source,
    files: Vec<RemoteInfo>,
    base_model: Option<String>,
}

#[derive(Clone, Debug)]
struct RemoteInfo {
    path: String,
    size: u64,
    sha256: Option<String>,
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
    let dir = models_root().join(&id);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|error| format!("create {}: {error}", dir.display()))?;
    let total = plan.files.iter().map(|file| file.size).sum::<u64>();
    if source == Source::ModelScope {
        let files = plan.files.clone();
        let target = dir.clone();
        let app_handle = app.clone();
        tokio::task::spawn_blocking(move || download_modelscope(&app_handle, &files, &target))
            .await
            .map_err(|error| format!("modelscope download task: {error}"))??;
    } else {
        let client = reqwest::Client::builder()
            .user_agent("UniInfer-Desktop/0.0")
            .build()
            .map_err(|error| format!("http client: {error}"))?;
        let mut done = 0u64;
        for file in &plan.files {
            let target = dir.join(&file.path);
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
    let directory = dir.clone();
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
    Ok(InstallResult {
        id,
        path: dir.display().to_string(),
        restart_required: false,
    })
}

struct PlannedFiles {
    #[cfg_attr(not(test), allow(dead_code))]
    format: String,
    files: Vec<RemoteFile>,
}

#[derive(Clone, Debug)]
struct VariantSpec {
    key: String,
    label: String,
    source: String,
    format: String,
    quantization: String,
    files: Vec<RemoteFile>,
    recommended: bool,
}

#[cfg(test)]
fn build_plan(
    repo_files: &RepoFiles,
    base_files: Option<&RepoFiles>,
) -> Result<PlannedFiles, String> {
    build_plan_with_variant(repo_files, base_files, None)
}

fn build_plan_with_variant(
    repo_files: &RepoFiles,
    base_files: Option<&RepoFiles>,
    variant: Option<&str>,
) -> Result<PlannedFiles, String> {
    let variants = build_variants(repo_files, base_files)?;
    let chosen = choose_variant(&variants, variant)?;
    Ok(PlannedFiles {
        format: chosen.format.clone(),
        files: chosen.files.clone(),
    })
}

fn choose_variant<'a>(
    variants: &'a [VariantSpec],
    key: Option<&str>,
) -> Result<&'a VariantSpec, String> {
    if let Some(key) = key.filter(|key| !key.is_empty()) {
        return variants
            .iter()
            .find(|variant| variant.key == key)
            .ok_or_else(|| format!("找不到下载变体：{key}"));
    }
    variants
        .iter()
        .find(|variant| variant.recommended)
        .or_else(|| variants.first())
        .ok_or_else(|| "没有可下载的模型文件".into())
}

fn build_variants(
    repo_files: &RepoFiles,
    base_files: Option<&RepoFiles>,
) -> Result<Vec<VariantSpec>, String> {
    let files = &repo_files.files;
    let has = |ext: &str| files.iter().any(|file| file.path.ends_with(ext));
    let format = if has(".gguf") {
        "gguf"
    } else if has(".safetensors") {
        "safetensors"
    } else if has(".onnx") {
        "onnx"
    } else {
        return Err("仓库里没有 GGUF、Safetensors 或 ONNX 模型文件".into());
    };

    let mut variants = Vec::new();
    match format {
        "gguf" => {
            let mut candidates = files
                .iter()
                .filter(|file| {
                    file.path.ends_with(".gguf") && !file.path.to_lowercase().contains("mmproj")
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                return Err("没有可用的 GGUF 量化文件".into());
            }
            candidates.sort_by(|a, b| {
                quant_rank(&a.path)
                    .cmp(&quant_rank(&b.path))
                    .then_with(|| a.path.cmp(&b.path))
            });
            for candidate in candidates {
                let mut selected = vec![remote_file(repo_files, candidate)];
                selected.extend(sidecars(repo_files));
                if !selected
                    .iter()
                    .any(|file| file.path.ends_with("tokenizer.json"))
                {
                    let base = base_files.ok_or_else(|| {
                        "GGUF 仓库没有 tokenizer.json，也没有可用的 base model".to_string()
                    })?;
                    selected.extend(sidecars(base));
                }
                dedup_files(&mut selected);
                let quantization = quant_label(&candidate.path);
                variants.push(VariantSpec {
                    key: format!("gguf:{}", candidate.path),
                    label: quantization.clone(),
                    source: candidate.path.clone(),
                    format: "gguf".into(),
                    quantization,
                    files: selected,
                    recommended: variants.is_empty(),
                });
            }
        }
        "safetensors" => {
            let mut selected = files
                .iter()
                .filter(|file| {
                    file.path.ends_with(".safetensors")
                        || file.path.ends_with(".safetensors.index.json")
                })
                .map(|file| remote_file(repo_files, file))
                .collect::<Vec<_>>();
            selected.extend(sidecars(repo_files));
            dedup_files(&mut selected);
            if selected.is_empty() {
                return Err("没有可用的 Safetensors 模型文件".into());
            }
            let source = selected
                .iter()
                .find(|file| file.path.ends_with(".safetensors"))
                .map(|file| file.path.clone())
                .unwrap_or_default();
            let quantization = quant_label(&source);
            variants.push(VariantSpec {
                key: "safetensors".into(),
                label: "Safetensors".into(),
                source,
                format: "safetensors".into(),
                quantization,
                files: selected,
                recommended: true,
            });
        }
        "onnx" => {
            let mut groups: std::collections::BTreeMap<String, Vec<&RemoteInfo>> =
                std::collections::BTreeMap::new();
            for file in files.iter().filter(|file| file.path.ends_with(".onnx")) {
                let dir = file
                    .path
                    .rsplit_once('/')
                    .map_or(String::new(), |(dir, _)| dir.to_string());
                groups.entry(dir).or_default().push(file);
            }
            let mut groups = groups.into_iter().collect::<Vec<_>>();
            groups.sort_by(|a, b| {
                let a_size = a.1.iter().map(|file| file.size).sum::<u64>();
                let b_size = b.1.iter().map(|file| file.size).sum::<u64>();
                a_size.cmp(&b_size).then_with(|| a.0.cmp(&b.0))
            });
            if groups.is_empty() {
                return Err("没有可用的 ONNX 模型文件".into());
            }
            for (dir, items) in groups {
                let mut selected = items
                    .into_iter()
                    .map(|file| remote_file(repo_files, file))
                    .collect::<Vec<_>>();
                selected.extend(sidecars(repo_files));
                dedup_files(&mut selected);
                let label = if dir.is_empty() {
                    "ONNX 默认".to_string()
                } else {
                    dir.clone()
                };
                let quantization = quant_label(&label);
                variants.push(VariantSpec {
                    key: format!("onnx:{dir}"),
                    label,
                    source: dir,
                    format: "onnx".into(),
                    quantization,
                    files: selected,
                    recommended: variants.is_empty(),
                });
            }
        }
        _ => unreachable!(),
    }
    if variants.is_empty() {
        return Err("没有可下载的模型文件".into());
    }
    Ok(variants)
}

fn dedup_files(files: &mut Vec<RemoteFile>) {
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files.dedup_by(|a, b| a.path == b.path);
}

fn quant_label(path: &str) -> String {
    let lower = path.to_lowercase();
    for (needle, label) in [
        ("q8_0", "Q8_0"),
        ("q6_k", "Q6_K"),
        ("q5_k_m", "Q5_K_M"),
        ("q5_k_s", "Q5_K_S"),
        ("q4_k_m", "Q4_K_M"),
        ("q4_k_s", "Q4_K_S"),
        ("q4_0", "Q4_0"),
        ("q3_k_m", "Q3_K_M"),
        ("q2_k", "Q2_K"),
        ("fp8", "FP8"),
        ("fp16", "FP16"),
        ("bf16", "BF16"),
        ("int8", "INT8"),
        ("int4", "INT4"),
    ] {
        if lower.contains(needle) {
            return label.into();
        }
    }
    "unknown".into()
}

fn remote_file(repo: &RepoFiles, info: &RemoteInfo) -> RemoteFile {
    RemoteFile {
        repo: repo.repo.clone(),
        path: info.path.clone(),
        url: file_url(repo.source, &repo.repo, &info.path),
        sha256: info.sha256.clone(),
        size: info.size,
    }
}

fn sidecars(repo: &RepoFiles) -> Vec<RemoteFile> {
    repo.files
        .iter()
        .filter(|file| is_sidecar(&file.path))
        .map(|file| remote_file(repo, file))
        .collect()
}

fn is_sidecar(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    matches!(
        name,
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "special_tokens_map.json"
            | "preprocessor_config.json"
            | "processor_config.json"
            | "vocab.json"
            | "merges.txt"
    )
}

fn quant_rank(path: &str) -> u8 {
    let lower = path.to_lowercase();
    if lower.contains("q4_k_m") {
        0
    } else if lower.contains("q4_k_s") {
        1
    } else if lower.contains("q4_0") {
        2
    } else if lower.contains("q5_k_m") {
        3
    } else if lower.contains("q5_k_s") {
        4
    } else if lower.contains("q8_0") {
        5
    } else if lower.contains("bf16") {
        6
    } else {
        10
    }
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

fn models_root() -> PathBuf {
    crate::api_server::models_root()
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

async fn search_huggingface(query: &str, page: u32, page_size: u32) -> Result<SearchPage, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(25))
        .user_agent("UniInfer-Desktop/0.0")
        .build()
        .map_err(|error| format!("search client: {error}"))?;
    let mut last_error = String::new();
    let offset = (page - 1) * page_size;
    let fetch = offset + page_size + 1;
    for base in ["https://hf-mirror.com", "https://huggingface.co"] {
        let response = client
            .get(format!("{base}/api/models"))
            .header("Accept", "application/json")
            .query(&[
                ("search", query.to_string()),
                ("limit", fetch.to_string()),
                ("sort", "downloads".to_string()),
                ("direction", "-1".to_string()),
            ])
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                last_error = format!("{base}: {error}");
                continue;
            }
        };
        if !response.status().is_success() {
            last_error = format!("{base}: HTTP {}", response.status());
            continue;
        }
        let items: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|error| format!("search huggingface: {error}"))?;
        let has_more = items.len() > (offset + page_size) as usize;
        let items = items
            .into_iter()
            .skip(offset as usize)
            .take(page_size as usize)
            .filter_map(|item| {
                let id = item.get("id")?.as_str()?.to_string();
                Some(SearchResult {
                    id,
                    downloads: item
                        .get("downloads")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0),
                    task: item
                        .get("pipeline_tag")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    tags: item
                        .get("tags")
                        .and_then(serde_json::Value::as_array)
                        .map(|tags| {
                            tags.iter()
                                .filter_map(|tag| tag.as_str().map(str::to_string))
                                .take(4)
                                .collect()
                        })
                        .unwrap_or_default(),
                })
            })
            .collect();
        return Ok(SearchPage {
            items,
            page,
            page_size,
            has_more,
            total: None,
        });
    }
    Err(format!("Hugging Face 搜索失败：{last_error}"))
}

async fn search_modelscope(query: &str, page: u32, page_size: u32) -> Result<SearchPage, String> {
    let response = reqwest::Client::new()
        .get("https://modelscope.cn/openapi/v1/models")
        .query(&[
            ("page_number", page.to_string()),
            ("page_size", page_size.to_string()),
            ("search", query.to_string()),
        ])
        .send()
        .await
        .map_err(|error| format!("search modelscope: {error}"))?
        .error_for_status()
        .map_err(|error| format!("search modelscope: {error}"))?;
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("search modelscope: {error}"))?;
    if value.get("success").and_then(serde_json::Value::as_bool) == Some(false) {
        return Err("ModelScope 搜索失败".into());
    }
    let items = value
        .pointer("/data/models")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let total = value
        .pointer("/data/total_count")
        .and_then(serde_json::Value::as_u64);
    let items: Vec<SearchResult> = items
        .into_iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?.to_string();
            Some(SearchResult {
                id,
                downloads: item
                    .get("downloads")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                task: item
                    .get("tasks")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|tasks| tasks.first())
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                tags: item
                    .get("tags")
                    .and_then(serde_json::Value::as_array)
                    .map(|tags| {
                        tags.iter()
                            .filter_map(|tag| tag.as_str().map(str::to_string))
                            .take(4)
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        })
        .collect();
    let has_more = total.map_or(items.len() >= page_size as usize, |total| {
        (page as u64) * (page_size as u64) < total
    });
    Ok(SearchPage {
        items,
        page,
        page_size,
        has_more,
        total,
    })
}

async fn fetch_repo_files(
    repo: &str,
    source: Source,
) -> Result<(RepoFiles, Option<RepoFiles>), String> {
    let repo_files = match source {
        Source::HuggingFace => fetch_hf_repo(repo).await?,
        Source::ModelScope => fetch_modelscope_repo(repo).await?,
    };
    let base_files = match &repo_files.base_model {
        Some(base) => Some(match source {
            Source::HuggingFace => fetch_hf_repo(base).await?,
            Source::ModelScope => fetch_modelscope_repo(base).await?,
        }),
        None => None,
    };
    Ok((repo_files, base_files))
}

async fn fetch_hf_repo(repo: &str) -> Result<RepoFiles, String> {
    let value = reqwest::get(format!(
        "https://hf-mirror.com/api/models/{repo}?blobs=true"
    ))
    .await
    .map_err(|error| format!("repo metadata: {error}"))?
    .error_for_status()
    .map_err(|error| format!("repo metadata: {error}"))?
    .json::<serde_json::Value>()
    .await
    .map_err(|error| format!("repo metadata: {error}"))?;
    let files = value
        .get("siblings")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let path = item.get("rfilename")?.as_str()?.to_string();
            let lfs = item.get("lfs");
            let size = lfs
                .and_then(|lfs| lfs.get("size"))
                .and_then(serde_json::Value::as_u64)
                .or_else(|| item.get("size").and_then(serde_json::Value::as_u64))
                .unwrap_or(0);
            let sha256 = lfs
                .and_then(|lfs| lfs.get("oid"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            Some(RemoteInfo { path, size, sha256 })
        })
        .collect::<Vec<_>>();
    let base_model = value
        .pointer("/cardData/base_model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Ok(RepoFiles {
        repo: repo.to_string(),
        source: Source::HuggingFace,
        files,
        base_model,
    })
}

async fn fetch_modelscope_repo(repo: &str) -> Result<RepoFiles, String> {
    let value = reqwest::get(format!(
        "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Root="
    ))
    .await
    .map_err(|error| format!("repo files: {error}"))?
    .error_for_status()
    .map_err(|error| format!("repo files: {error}"))?
    .json::<serde_json::Value>()
    .await
    .map_err(|error| format!("repo files: {error}"))?;
    let files = value
        .pointer("/Data/Files")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let path = item.get("Path")?.as_str()?.to_string();
            let size = item
                .get("Size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let sha256 = item
                .get("Sha256")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            Some(RemoteInfo { path, size, sha256 })
        })
        .collect::<Vec<_>>();
    let detail = match reqwest::get(format!("https://modelscope.cn/api/v1/models/{repo}")).await {
        Ok(response) => response.json::<serde_json::Value>().await.ok(),
        Err(_) => None,
    };
    let base_model = detail
        .as_ref()
        .and_then(|value| value.pointer("/Data/BaseModel/0"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Ok(RepoFiles {
        repo: repo.to_string(),
        source: Source::ModelScope,
        files,
        base_model,
    })
}

fn file_url(source: Source, repo: &str, path: &str) -> String {
    let path = path
        .split('/')
        .map(|segment| urlencoding::encode(segment).into_owned())
        .collect::<Vec<_>>()
        .join("/");
    match source {
        Source::HuggingFace => format!("https://hf-mirror.com/{repo}/resolve/main/{path}"),
        Source::ModelScope => {
            format!("https://modelscope.cn/models/{repo}/resolve/master/{path}")
        }
    }
}

#[cfg(test)]
mod tests {
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
    fn slug_is_filesystem_safe() {
        assert_eq!(slug("Qwen/Qwen3-0.6B"), "Qwen-Qwen3-0.6B");
    }
}
