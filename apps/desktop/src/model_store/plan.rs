//! Download planning and variant selection for online models.

use super::RemoteFile;
use super::repo::{RemoteInfo, RepoFiles, file_url};

#[cfg(test)]
pub(super) struct PlannedFiles {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) format: String,
    pub(super) files: Vec<RemoteFile>,
}

#[derive(Clone, Debug)]
pub(super) struct VariantSpec {
    pub(super) key: String,
    pub(super) label: String,
    pub(super) source: String,
    pub(super) format: String,
    pub(super) quantization: String,
    pub(super) files: Vec<RemoteFile>,
    pub(super) recommended: bool,
}

#[cfg(test)]
pub(super) fn build_plan(
    repo_files: &RepoFiles,
    base_files: Option<&RepoFiles>,
) -> Result<PlannedFiles, String> {
    build_plan_with_variant(repo_files, base_files, None)
}

#[cfg(test)]
pub(super) fn build_plan_with_variant(
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

pub(super) fn choose_variant<'a>(
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

pub(super) fn build_variants(
    repo_files: &RepoFiles,
    base_files: Option<&RepoFiles>,
) -> Result<Vec<VariantSpec>, String> {
    let files = &repo_files.files;
    let mut variants = Vec::new();

    let mut gguf = files
        .iter()
        .filter(|file| {
            file.path.to_ascii_lowercase().ends_with(".gguf")
                && !file.path.to_lowercase().contains("mmproj")
        })
        .collect::<Vec<_>>();
    if !gguf.is_empty() {
        gguf.sort_by(|a, b| {
            quant_rank(&a.path)
                .cmp(&quant_rank(&b.path))
                .then_with(|| a.path.cmp(&b.path))
        });
        for candidate in gguf {
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
                recommended: false,
            });
        }
    }

    if files
        .iter()
        .any(|file| file.path.to_ascii_lowercase().ends_with(".safetensors"))
    {
        let mut selected = files
            .iter()
            .filter(|file| {
                let path = file.path.to_ascii_lowercase();
                path.ends_with(".safetensors") || path.ends_with(".safetensors.index.json")
            })
            .map(|file| remote_file(repo_files, file))
            .collect::<Vec<_>>();
        selected.extend(sidecars(repo_files));
        dedup_files(&mut selected);
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
            recommended: false,
        });
    }

    if is_diffusion_repo(files) {
        let selected = files
            .iter()
            .filter(|file| is_diffusion_asset(&file.path))
            .map(|file| remote_file(repo_files, file))
            .collect::<Vec<_>>();
        variants.push(VariantSpec {
            key: "onnx:components".into(),
            label: "ONNX 完整模型".into(),
            source: "全部组件".into(),
            format: "onnx".into(),
            quantization: "unknown".into(),
            files: selected,
            recommended: false,
        });
    } else {
        let mut groups: std::collections::BTreeMap<(String, String), Vec<&RemoteInfo>> =
            std::collections::BTreeMap::new();
        for file in files
            .iter()
            .filter(|file| file.path.to_ascii_lowercase().ends_with(".onnx"))
        {
            let dir = file
                .path
                .rsplit_once('/')
                .map_or(String::new(), |(dir, _)| dir.to_string());
            let quant = onnx_quant_suffix(&file.path);
            groups.entry((dir, quant)).or_default().push(file);
        }
        if !groups.is_empty() {
            let mut groups = groups.into_iter().collect::<Vec<_>>();
            groups.sort_by(|a, b| {
                let a_size = a.1.iter().map(|file| file.size).sum::<u64>();
                let b_size = b.1.iter().map(|file| file.size).sum::<u64>();
                a_size.cmp(&b_size).then_with(|| a.0.cmp(&b.0))
            });
            for ((dir, quant), items) in groups {
                let mut selected = items
                    .iter()
                    .map(|file| remote_file(repo_files, file))
                    .collect::<Vec<_>>();
                for data in files
                    .iter()
                    .filter(|file| is_onnx_external_data(&file.path))
                {
                    if items.iter().any(|graph| data.path.starts_with(&graph.path)) {
                        selected.push(remote_file(repo_files, data));
                    }
                }
                selected.extend(sidecars(repo_files));
                dedup_files(&mut selected);
                let label = if quant.is_empty() {
                    if dir.is_empty() {
                        "ONNX 默认".to_string()
                    } else {
                        dir.clone()
                    }
                } else if dir.is_empty() {
                    format!("ONNX {quant}")
                } else {
                    format!("{dir} · {quant}")
                };
                let quantization = quant_label(&label);
                variants.push(VariantSpec {
                    key: format!("onnx:{dir}:{quant}"),
                    label,
                    source: dir,
                    format: "onnx".into(),
                    quantization,
                    files: selected,
                    recommended: false,
                });
            }
        }
    }

    if variants.is_empty() {
        return Err("仓库里没有 GGUF、Safetensors 或 ONNX 模型文件".into());
    }
    let preferred = ["gguf", "safetensors", "onnx"]
        .into_iter()
        .find(|format| variants.iter().any(|variant| variant.format == *format));
    if let Some(preferred) = preferred
        && let Some(variant) = variants
            .iter_mut()
            .find(|variant| variant.format == preferred)
    {
        variant.recommended = true;
    }
    Ok(variants)
}

pub(super) fn dedup_files(files: &mut Vec<RemoteFile>) {
    files.sort_by(|a, b| a.path.cmp(&b.path));
    files.dedup_by(|a, b| a.path == b.path);
}

const DIFFUSION_COMPONENT_ROOTS: &[&str] = &[
    "text_encoder",
    "text_encoder_2",
    "unet",
    "transformer",
    "vae",
    "vae_decoder",
    "vae_encoder",
];

fn is_diffusion_repo(files: &[RemoteInfo]) -> bool {
    let roots = files
        .iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".onnx"))
        .filter_map(|file| {
            file.path
                .split_once('/')
                .map(|(root, _)| root.to_ascii_lowercase())
        })
        .filter(|root| DIFFUSION_COMPONENT_ROOTS.contains(&root.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    roots.len() >= 2
        && roots.iter().any(|root| {
            matches!(
                root.as_str(),
                "unet" | "transformer" | "vae" | "vae_decoder" | "vae_encoder"
            )
        })
}

fn is_diffusion_asset(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    !name.starts_with('.')
        && name != "readme.md"
        && !name.starts_with("license")
        && !path
            .split('/')
            .any(|segment| segment == ".git" || segment.starts_with('.'))
}

pub(super) fn is_onnx_external_data(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    name.contains(".onnx_data") || name.ends_with(".onnx.data")
}

pub(super) fn onnx_quant_suffix(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.split(".onnx").next().unwrap_or(name);
    let lower = stem.to_ascii_lowercase();
    for suffix in [
        "q4f16",
        "q4",
        "q8",
        "fp16",
        "fp32",
        "int8",
        "uint8",
        "quantized",
        "bnb4",
    ] {
        if lower.ends_with(&format!("_{suffix}")) {
            return suffix.to_string();
        }
    }
    String::new()
}

pub(super) fn quant_label(path: &str) -> String {
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

pub(super) fn remote_file(repo: &RepoFiles, info: &RemoteInfo) -> RemoteFile {
    RemoteFile {
        path: info.path.clone(),
        url: file_url(repo.source, &repo.repo, &info.path),
        sha256: info.sha256.clone(),
        size: info.size,
    }
}

pub(super) fn sidecars(repo: &RepoFiles) -> Vec<RemoteFile> {
    repo.files
        .iter()
        .filter(|file| is_sidecar(&file.path))
        .map(|file| remote_file(repo, file))
        .collect()
}

pub(super) fn is_sidecar(path: &str) -> bool {
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

pub(super) fn quant_rank(path: &str) -> u8 {
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
