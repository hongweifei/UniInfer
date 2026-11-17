//! Repository file metadata for Hugging Face and ModelScope.

use super::Source;

#[derive(Clone, Debug)]
pub(super) struct RepoFiles {
    pub(super) repo: String,
    pub(super) source: Source,
    pub(super) files: Vec<RemoteInfo>,
    pub(super) base_model: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct RemoteInfo {
    pub(super) path: String,
    pub(super) size: u64,
    pub(super) sha256: Option<String>,
}

pub(super) async fn fetch_repo_files(
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

pub(super) async fn fetch_hf_repo(repo: &str) -> Result<RepoFiles, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("UniInfer-Desktop/0.0")
        .build()
        .map_err(|error| format!("repo client: {error}"))?;
    let mut last_error = String::new();
    let mut value = None;
    for base in ["https://hf-mirror.com", "https://huggingface.co"] {
        match client
            .get(format!("{base}/api/models/{repo}?blobs=true"))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(parsed) => {
                        value = Some(parsed);
                        break;
                    }
                    Err(error) => last_error = format!("{base}: {error}"),
                }
            }
            Ok(response) => last_error = format!("{base}: HTTP {}", response.status()),
            Err(error) => last_error = format!("{base}: {error}"),
        }
    }
    let value = value.ok_or_else(|| format!("repo metadata: {last_error}"))?;
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

fn modelscope_repo_error(repo: &str, status: reqwest::StatusCode) -> String {
    if status == reqwest::StatusCode::NOT_FOUND {
        format!(
            "ModelScope 上不存在仓库 {repo}（HTTP 404）；如果它来自 Hugging Face，请切换模型源后重试"
        )
    } else {
        format!("ModelScope 仓库 {repo} 请求失败：HTTP {status}")
    }
}

pub(super) async fn fetch_modelscope_repo(repo: &str) -> Result<RepoFiles, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("UniInfer-Desktop/0.0")
        .build()
        .map_err(|error| format!("repo client: {error}"))?;
    let response = client
        .get(format!(
            "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Root=&Recursive=true"
        ))
        .send()
        .await
        .map_err(|error| format!("repo files: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(modelscope_repo_error(repo, status));
    }
    let value = response
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
    let detail = match client
        .get(format!("https://modelscope.cn/api/v1/models/{repo}"))
        .send()
        .await
    {
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

pub(super) fn file_url(source: Source, repo: &str, path: &str) -> String {
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
