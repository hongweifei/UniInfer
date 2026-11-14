#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api_server;

use serde::Serialize;
use tauri::Manager;

#[derive(Clone, Serialize)]
struct DesktopStatus {
    info: Option<api_server::ApiInfo>,
    error: Option<String>,
    manifest: String,
}

struct DesktopState {
    status: DesktopStatus,
    _runtime: Option<std::sync::Arc<tokio::runtime::Runtime>>,
}

#[tauri::command]
fn api_info(state: tauri::State<'_, DesktopState>) -> DesktopStatus {
    state.status.clone()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![api_info])
        .setup(|app| {
            let startup = api_server::start();
            app.manage(DesktopState {
                status: DesktopStatus {
                    info: startup.info,
                    error: startup.error,
                    manifest: api_server::manifest_display(),
                },
                _runtime: startup.runtime,
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run UniInfer desktop");
}
