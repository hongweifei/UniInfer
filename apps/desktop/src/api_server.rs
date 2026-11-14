use std::sync::Arc;

use serde::Serialize;

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

pub(crate) fn manifest_path() -> String {
    std::env::var("UNIINFER_MANIFEST").unwrap_or_else(|_| "models/tiny-onnx/manifest.toml".into())
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
    let token = random_token();
    let config = uniinfer_api::ApiConfig::new(manifest_path().into()).with_token(token.clone());
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
