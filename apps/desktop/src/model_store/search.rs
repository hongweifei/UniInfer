//! Online model search transports.

use super::{SearchPage, SearchResult};

pub(super) async fn search_huggingface(
    query: &str,
    page: u32,
    page_size: u32,
) -> Result<SearchPage, String> {
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

pub(super) async fn search_modelscope(
    query: &str,
    page: u32,
    page_size: u32,
) -> Result<SearchPage, String> {
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
