#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod api_server;
mod backend_store;
mod model_store;

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

#[tauri::command]
fn window_start_dragging(window: tauri::WebviewWindow) {
    let _ = window.start_dragging();
}

#[tauri::command]
fn window_minimize(window: tauri::WebviewWindow) {
    let _ = window.minimize();
}

#[tauri::command]
fn window_toggle_maximize(window: tauri::WebviewWindow) {
    if window.is_maximized().unwrap_or(false) {
        let _ = window.unmaximize();
    } else {
        let _ = window.maximize();
    }
}

#[tauri::command]
fn window_close(window: tauri::WebviewWindow) {
    let _ = window.close();
}

#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    app.restart();
}

#[tauri::command]
fn set_window_theme(window: tauri::WebviewWindow, theme: String) {
    let theme = match theme.as_str() {
        "light" => Some(tauri::Theme::Light),
        "dark" => Some(tauri::Theme::Dark),
        _ => None,
    };
    let _ = window.set_theme(theme);
}

#[tauri::command]
async fn pick_model_folder() -> Option<String> {
    tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("选择模型目录")
            .pick_folder()
            .map(|path| path.display().to_string())
    })
    .await
    .ok()
    .flatten()
}

#[tauri::command]
async fn pick_backend_archive() -> Option<String> {
    tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("选择后端包 zip")
            .add_filter("后端包", &["zip"])
            .pick_file()
            .map(|path| path.display().to_string())
    })
    .await
    .ok()
    .flatten()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            api_info,
            restart_app,
            window_start_dragging,
            window_minimize,
            window_toggle_maximize,
            window_close,
            set_window_theme,
            pick_model_folder,
            pick_backend_archive,
            model_store::search_models,
            model_store::plan_model,
            model_store::download::start_download,
            model_store::download::start_backend_download,
            model_store::download::list_downloads,
            model_store::download::cancel_download,
            model_store::download::remove_download,
            model_store::download::clear_downloads,
            model_store::delete_model,
            backend_store::fetch_backend_catalog,
            backend_store::import_backend_archive
        ])
        .setup(|app| {
            let startup = api_server::start();
            app.manage(model_store::DownloadManager::new());
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
