use std::sync::Arc;

use serde::Serialize;
use uniinfer_config::{ConfigHandle, ModelEntry, UniInferConfig};

#[derive(Clone, Serialize)]
pub(crate) struct ApiInfo {
    pub(crate) base: String,
    pub(crate) token: String,
}

pub(crate) struct ApiStartup {
    pub(crate) info: Option<ApiInfo>,
    pub(crate) runtime: Option<Arc<tokio::runtime::Runtime>>,
    pub(crate) error: Option<String>,
}

pub(crate) fn manifest_paths() -> Vec<std::path::PathBuf> {
    if let Ok(list) = std::env::var("UNIINFER_MANIFESTS") {
        let paths = list
            .split(';')
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>();
        if !paths.is_empty() {
            return paths;
        }
    }
    vec![
        std::env::var("UNIINFER_MANIFEST")
            .unwrap_or_else(|_| "models/tiny-onnx/manifest.toml".into())
            .into(),
    ]
}

pub(crate) fn manifest_display() -> String {
    manifest_paths()
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn start() -> ApiStartup {
    match start_inner() {
        Ok((info, runtime)) => ApiStartup {
            info: Some(info),
            runtime: Some(runtime),
            error: None,
        },
        Err(error) => ApiStartup {
            info: None,
            runtime: None,
            error: Some(error.to_string()),
        },
    }
}

fn start_inner() -> Result<(ApiInfo, Arc<tokio::runtime::Runtime>), Box<dyn std::error::Error>> {
    let config_path = std::env::var_os("UNIINFER_CONFIG").map(std::path::PathBuf::from);
    let handle = if let Some(path) = config_path {
        ConfigHandle::load(path)?
    } else {
        let mut config = UniInferConfig::default();
        config.models = manifest_paths()
            .into_iter()
            .map(ModelEntry::new)
            .collect::<Vec<_>>();
        ConfigHandle::new(config, Some(std::path::PathBuf::from("uniinfer.toml")))
    };
    let token = handle.snapshot()?.server.token.unwrap_or_else(random_token);
    let config = uniinfer_api::ApiConfig::from_config(Arc::new(handle))?.with_token(token.clone());
    let state = uniinfer_api::AppState::new(&config)?;
    let app = uniinfer_api::routes::router(state);

    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let addr = listener.local_addr()?;
    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?,
    );

    runtime.spawn(async move {
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => listener,
            Err(error) => {
                eprintln!("api listener: {error}");
                return;
            }
        };
        if let Err(error) = uniinfer_api::serve_listener(listener, app).await {
            eprintln!("api server: {error}");
        }
    });

    Ok((
        ApiInfo {
            base: format!("http://{addr}"),
            token,
        },
        runtime,
    ))
}

fn random_token() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("system randomness");
    hex::encode(bytes)
}
