use std::time::Duration;

use super::variants::has_windows_x64_assets;
use super::{GITHUB_MIRROR, GithubAsset, GithubRelease};

const ASSET_MIRRORS: &[&str] = &[
    "https://ghfast.top/",
    "https://ghproxy.net/",
    "https://gh-proxy.com/",
];

pub(super) async fn github_releases(
    client: &reqwest::Client,
    repo: &str,
) -> Result<(Vec<GithubRelease>, bool), String> {
    let direct = format!("https://api.github.com/repos/{repo}/releases?per_page=20");
    let api_error = match fetch_releases(client, &direct).await {
        Ok(releases) => return Ok((releases, false)),
        Err(error) => format!("直连：{error}"),
    };
    let mirrored = format!("{GITHUB_MIRROR}{direct}");
    let api_error = match fetch_releases(client, &mirrored).await {
        Ok(releases) => return Ok((releases, true)),
        Err(error) => format!("{api_error}；镜像：{error}"),
    };
    github_atom_releases(client, repo)
        .await
        .map_err(|atom_error| {
            format!(
                "无法访问 GitHub（{api_error}；Atom：{atom_error}）。请检查网络或代理，也可以稍后重试。"
            )
        })
}

async fn fetch_releases(client: &reqwest::Client, url: &str) -> Result<Vec<GithubRelease>, String> {
    let response = client
        .get(url)
        .timeout(Duration::from_secs(90))
        .header(reqwest::header::USER_AGENT, "UniInfer")
        .send()
        .await
        .map_err(|error| format!("{error}"))?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    response
        .json()
        .await
        .map_err(|error| format!("解析响应失败：{error}"))
}

async fn github_atom_releases(
    client: &reqwest::Client,
    repo: &str,
) -> Result<(Vec<GithubRelease>, bool), String> {
    let direct = format!("https://github.com/{repo}/releases.atom");
    let (body, mirrored) = match fetch_text(client, &direct).await {
        Ok(body) => (body, false),
        Err(direct_error) => {
            let mirrored = format!("{GITHUB_MIRROR}{direct}");
            let body = fetch_text(client, &mirrored)
                .await
                .map_err(|mirror_error| format!("直连：{direct_error}；镜像：{mirror_error}"))?;
            (body, true)
        }
    };
    let releases = parse_atom_releases(&body, repo);
    if releases.is_empty() {
        return Err("Atom feed 没有可解析的 release".into());
    }
    // Some repositories publish release notes without the asset list. The
    // expanded assets fragment has the same files without using the GitHub API.
    // Stop at the newest release that has a Windows x64 binary.
    for mut release in releases {
        if release.assets.is_empty()
            && let Ok(assets) = fetch_expanded_assets(client, repo, &release.tag_name).await
        {
            release.assets = assets;
        }
        if has_windows_x64_assets(&release.assets) {
            return Ok((vec![release], mirrored));
        }
    }
    Err("Atom feed 没有可解析的 Windows release 资产".into())
}

async fn fetch_expanded_assets(
    client: &reqwest::Client,
    repo: &str,
    tag: &str,
) -> Result<Vec<GithubAsset>, String> {
    let path = format!(
        "https://github.com/{repo}/releases/expanded_assets/{}",
        tag.trim()
    );
    let mut last_error = String::new();
    for mirror in ASSET_MIRRORS {
        let url = format!("{mirror}{path}");
        match fetch_text(client, &url).await {
            Ok(body) => {
                let assets = parse_expanded_assets(&body, repo, tag);
                if !assets.is_empty() {
                    return Ok(assets);
                }
                last_error = "资产列表为空".into();
            }
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

pub(super) fn parse_expanded_assets(body: &str, repo: &str, tag: &str) -> Vec<GithubAsset> {
    let needle = format!("/{repo}/releases/download/{tag}/");
    let mut assets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for block in body.split("<li ").skip(1) {
        let Some(href) = between(block, "href=\"", "\"") else {
            continue;
        };
        if !href.contains(&needle) || !href.ends_with(".zip") {
            continue;
        }
        let name = href.rsplit('/').next().unwrap_or_default().to_owned();
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        let digest = block.find("sha256:").and_then(|index| {
            block[index + 7..]
                .get(..64)
                .filter(|value| value.chars().all(|character| character.is_ascii_hexdigit()))
                .map(|value| format!("sha256:{value}"))
        });
        assets.push(GithubAsset {
            name,
            browser_download_url: format!("https://github.com{href}"),
            size: 0,
            digest,
        });
    }
    assets
}

async fn fetch_text(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let response = client
        .get(url)
        .timeout(Duration::from_secs(90))
        .header(reqwest::header::USER_AGENT, "UniInfer")
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    response.text().await.map_err(|error| error.to_string())
}

pub(super) fn parse_atom_releases(body: &str, repo: &str) -> Vec<GithubRelease> {
    let mut releases = Vec::new();
    for entry in body.split("<entry>").skip(1) {
        let Some(end) = entry.find("</entry>") else {
            continue;
        };
        let entry = &entry[..end];
        let Some(tag) = between(entry, "<title>", "</title>") else {
            continue;
        };
        let tag = tag.trim();
        if tag.is_empty() {
            continue;
        }
        let prefix = format!("https://github.com/{repo}/releases/download/{tag}/");
        let mut assets = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut rest = entry;
        while let Some(index) = rest.find(&prefix) {
            let candidate = &rest[index..];
            let end = candidate
                .find(|character: char| {
                    character.is_whitespace() || matches!(character, '"' | '<' | '>' | '&')
                })
                .unwrap_or(candidate.len());
            let url = &candidate[..end];
            let name = url.rsplit('/').next().unwrap_or_default();
            if name.ends_with(".zip") && seen.insert(name.to_owned()) {
                assets.push(GithubAsset {
                    name: name.to_owned(),
                    browser_download_url: url.to_owned(),
                    size: 0,
                    digest: None,
                });
            }
            rest = &candidate[end..];
        }
        releases.push(GithubRelease {
            tag_name: tag.to_owned(),
            html_url: format!("https://github.com/{repo}/releases/tag/{tag}"),
            assets,
        });
    }
    releases
}

fn between<'a>(value: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let from = value.find(start)? + start.len();
    let to = value[from..].find(end)? + from;
    Some(&value[from..to])
}
