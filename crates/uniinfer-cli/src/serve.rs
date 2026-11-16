use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use uniinfer_api::ApiConfig;
use uniinfer_config::{ConfigHandle, UniInferConfig};

pub(crate) async fn serve(
    config_path: Option<PathBuf>,
    manifests: Vec<PathBuf>,
    addr: Option<std::net::SocketAddr>,
    queue_timeout_ms: Option<u64>,
    token: Option<String>,
) -> Result<()> {
    let source = match &config_path {
        Some(path) => ConfigHandle::load(path.clone())?,
        None => ConfigHandle::new(UniInferConfig::default(), None),
    };
    let mut config = source.snapshot()?;
    config.apply_env()?;
    let manifests = if manifests.is_empty() {
        discover_manifests(&std::path::PathBuf::from("models"))
    } else {
        manifests
    };
    if let Some(addr) = addr {
        config.server.addr = addr;
    }
    if let Some(queue_timeout_ms) = queue_timeout_ms {
        config.server.queue_timeout_ms = queue_timeout_ms;
    }
    if let Some(token) = token {
        config.server.token = Some(token);
    }
    config.validate_for_serve()?;
    let addr = config.server.addr;
    let handle = Arc::new(ConfigHandle::new(config, config_path));
    let api = ApiConfig::from_config(handle)?.with_manifests(manifests);
    uniinfer_api::serve(api, addr)
        .await
        .context("serve HTTP API")
}

fn discover_manifests(root: &std::path::Path) -> Vec<PathBuf> {
    fn walk(dir: &std::path::Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > 3 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, depth + 1, out);
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("manifest") && name.ends_with(".toml"))
            {
                out.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    walk(root, 0, &mut paths);
    paths.sort();
    paths
}
