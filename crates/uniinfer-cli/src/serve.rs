use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use uniinfer_api::ApiConfig;
use uniinfer_config::{ConfigHandle, ModelEntry, UniInferConfig};

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
    if !manifests.is_empty() {
        config.models = manifests.into_iter().map(ModelEntry::new).collect();
    }
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
    let api = ApiConfig::from_config(handle)?;
    uniinfer_api::serve(api, addr)
        .await
        .context("serve HTTP API")
}
