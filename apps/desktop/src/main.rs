#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use serde::Serialize;
use tauri::Manager;

#[derive(Clone, Serialize)]
struct ApiInfo {
    base: String,
    token: String,
}

#[derive(Clone, Serialize)]
struct DesktopStatus {
    info: Option<ApiInfo>,
    error: Option<String>,
    manifest: String,
}

struct DesktopState {
    status: DesktopStatus,
    _runtime: Option<Arc<tokio::runtime::Runtime>>,
}

#[tauri::command]
fn api_info(state: tauri::State<'_, DesktopState>) -> DesktopStatus {
    state.status.clone()
}

fn random_token() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("system randomness");
    hex::encode(bytes)
}

fn manifest_path() -> String {
    std::env::var("UNIINFER_MANIFEST").unwrap_or_else(|_| "models/tiny-onnx/manifest.toml".into())
}

fn start_api() -> Result<(ApiInfo, Arc<tokio::runtime::Runtime>), Box<dyn std::error::Error>> {
    let manifest = manifest_path();
    let token = random_token();
    let config = uniinfer_api::ApiConfig::new(manifest.into()).with_token(token.clone());
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

    let task_runtime = Arc::clone(&runtime);
    task_runtime.spawn(async move {
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

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![api_info])
        .setup(|app| {
            let (info, runtime, error) = match start_api() {
                Ok((info, runtime)) => (Some(info), Some(runtime), None),
                Err(error) => (None, None, Some(error.to_string())),
            };
            app.manage(DesktopState {
                status: DesktopStatus {
                    info,
                    error,
                    manifest: manifest_path(),
                },
                _runtime: runtime,
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run UniInfer desktop");
}
