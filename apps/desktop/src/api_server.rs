use std::sync::Arc;

use serde::Serialize;
use uniinfer_config::{ConfigHandle, UniInferConfig};

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
    discover_manifests()
}

pub(crate) fn project_root() -> Option<std::path::PathBuf> {
    roots()
        .into_iter()
        .find(|root| root.join("models").is_dir())
}

pub(crate) fn models_root() -> std::path::PathBuf {
    project_root()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("models")
}

fn roots() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let mut current = Some(dir);
        for _ in 0..6 {
            let Some(path) = current else { break };
            roots.push(path.to_path_buf());
            current = path.parent();
        }
    }
    roots.push(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
    roots.dedup();
    roots
}

fn discover_manifests() -> Vec<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, depth: usize, out: &mut Vec<std::path::PathBuf>) {
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
    for root in roots() {
        let models = root.join("models");
        if !models.is_dir() {
            continue;
        }
        let mut found = Vec::new();
        walk(&models, 0, &mut found);
        if !found.is_empty() {
            found.sort();
            return found;
        }
    }
    Vec::new()
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

pub(crate) fn backend_dir() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("UNIINFER_BACKEND_DIR") {
        return std::path::PathBuf::from(path);
    }
    if let Some(path) = config_path() {
        let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        if path.is_file()
            && let Ok(config) = UniInferConfig::load(&path)
            && let Some(dir) = config.server.backend_dir
        {
            return if dir.is_absolute() { dir } else { base.join(dir) };
        }
        return base.join("backends");
    }
    std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("backends")
}

fn config_path() -> Option<std::path::PathBuf> {
    std::env::var_os("UNIINFER_CONFIG")
        .map(std::path::PathBuf::from)
        .or_else(|| project_root().map(|root| root.join("uniinfer.toml")))
}

fn start_inner() -> Result<(ApiInfo, Arc<tokio::runtime::Runtime>), Box<dyn std::error::Error>> {
    let config_path = config_path();
    let handle = match config_path {
        Some(path) if path.is_file() => ConfigHandle::load(path)?,
        Some(path) => ConfigHandle::new(UniInferConfig::default(), Some(path)),
        None => ConfigHandle::new(
            UniInferConfig::default(),
            Some(std::path::PathBuf::from("uniinfer.toml")),
        ),
    };
    let mut snapshot = handle.snapshot()?;
    let token = match snapshot.server.token.clone() {
        Some(token) => token,
        None => {
            let token = random_token();
            snapshot.server.token = Some(token.clone());
            handle.replace(snapshot.clone())?;
            token
        }
    };
    let bind_addr = snapshot.server.addr;
    let manifests = manifest_paths();
    let model_roots = vec![models_root()];
    let config = uniinfer_api::ApiConfig::from_config(Arc::new(handle))?
        .with_manifests(manifests)
        .with_model_roots(model_roots);
    let state = uniinfer_api::AppState::new(&config)?;
    let app = uniinfer_api::routes::router(state);

    let listener = std::net::TcpListener::bind(bind_addr)?;
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
