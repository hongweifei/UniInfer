//! Download queue, progress, cancellation and resumable HTTP downloads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backend_store::BackendManifest;

use super::{RemoteFile, Source, models_root, slug};

mod fetch;
mod install;

use install::install_model;

use fetch::InstallError;

const PARALLEL_DOWNLOADS: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DownloadStatus {
    Queued,
    Preparing,
    Downloading,
    Verifying,
    Inspecting,
    Committing,
    Done,
    Failed,
    Cancelled,
}

impl DownloadStatus {
    fn terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DownloadKind {
    Model,
    Backend,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DownloadTask {
    pub(crate) id: String,
    pub(crate) kind: DownloadKind,
    pub(crate) repo: String,
    pub(crate) source: Source,
    pub(crate) variant: Option<String>,
    pub(crate) variant_label: Option<String>,
    pub(crate) model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) backend: Option<BackendManifest>,
    pub(crate) status: DownloadStatus,
    pub(crate) file: Option<String>,
    pub(crate) downloaded: u64,
    pub(crate) total: u64,
    pub(crate) message: Option<String>,
    pub(crate) path: Option<String>,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
}

struct ManagerState {
    tasks: Vec<DownloadTask>,
    controls: HashMap<String, Arc<AtomicBool>>,
}

#[derive(Clone)]
pub(crate) struct DownloadManager {
    inner: Arc<Mutex<ManagerState>>,
    slots: Arc<Semaphore>,
}

impl DownloadManager {
    pub(crate) fn new() -> Self {
        cleanup_stale_staging();
        Self {
            inner: Arc::new(Mutex::new(ManagerState {
                tasks: Vec::new(),
                controls: HashMap::new(),
            })),
            slots: Arc::new(Semaphore::new(PARALLEL_DOWNLOADS)),
        }
    }

    fn enqueue(
        &self,
        repo: String,
        source: Source,
        variant: Option<String>,
    ) -> Result<DownloadTask, String> {
        let model_id = slug(&repo);
        if model_id.is_empty() {
            return Err("模型仓库名不能为空".into());
        }
        let mut inner = self.inner.lock().map_err(|_| "下载状态锁已损坏")?;
        if inner
            .tasks
            .iter()
            .any(|task| task.repo == repo && task.source == source && !task.status.terminal())
        {
            return Err("这个模型已经在下载队列中".into());
        }
        let now = now_ms();
        let task = DownloadTask {
            id: next_task_id(),
            kind: DownloadKind::Model,
            repo,
            source,
            variant,
            variant_label: None,
            model_id,
            backend: None,
            status: DownloadStatus::Queued,
            file: None,
            downloaded: 0,
            total: 0,
            message: Some("等待下载".into()),
            path: None,
            created_at: now,
            updated_at: now,
        };
        inner
            .controls
            .insert(task.id.clone(), Arc::new(AtomicBool::new(false)));
        inner.tasks.push(task.clone());
        Ok(task)
    }

    fn enqueue_backend(&self, package: BackendManifest) -> Result<DownloadTask, String> {
        let mut inner = self.inner.lock().map_err(|_| "下载状态锁已损坏")?;
        if inner.tasks.iter().any(|task| {
            task.kind == DownloadKind::Backend
                && task.model_id == package.id
                && !task.status.terminal()
        }) {
            return Err("这个后端变体已经在下载队列中".into());
        }
        let variant = package
            .engines
            .first()
            .map(|engine| engine.variant.trim())
            .filter(|variant| !variant.is_empty())
            .unwrap_or("auto")
            .to_owned();
        let total = package.files.iter().map(|file| file.size).sum::<u64>();
        let now = now_ms();
        let task = DownloadTask {
            id: next_task_id(),
            kind: DownloadKind::Backend,
            repo: package.name.clone(),
            source: Source::Backend,
            variant: Some(package.id.clone()),
            variant_label: Some(variant),
            model_id: package.id.clone(),
            backend: Some(package),
            status: DownloadStatus::Queued,
            file: None,
            downloaded: 0,
            total,
            message: Some("等待下载".into()),
            path: None,
            created_at: now,
            updated_at: now,
        };
        inner
            .controls
            .insert(task.id.clone(), Arc::new(AtomicBool::new(false)));
        inner.tasks.push(task.clone());
        Ok(task)
    }

    pub(crate) fn list(&self) -> Vec<DownloadTask> {
        self.inner
            .lock()
            .map(|inner| inner.tasks.iter().rev().cloned().collect())
            .unwrap_or_default()
    }

    fn get(&self, id: &str) -> Option<DownloadTask> {
        self.inner
            .lock()
            .ok()?
            .tasks
            .iter()
            .find(|task| task.id == id)
            .cloned()
    }

    fn update<F>(&self, id: &str, update: F) -> Option<DownloadTask>
    where
        F: FnOnce(&mut DownloadTask),
    {
        let mut inner = self.inner.lock().ok()?;
        let cancelled = inner
            .controls
            .get(id)
            .is_some_and(|flag| flag.load(Ordering::Relaxed));
        let task = inner.tasks.iter_mut().find(|task| task.id == id)?;
        if task.status.terminal() || cancelled {
            return Some(task.clone());
        }
        update(task);
        task.updated_at = now_ms();
        Some(task.clone())
    }

    fn cancel(&self, id: &str) -> Result<DownloadTask, String> {
        let mut inner = self.inner.lock().map_err(|_| "下载状态锁已损坏")?;
        let cancelled = inner.controls.get(id).cloned();
        let task = inner
            .tasks
            .iter_mut()
            .find(|task| task.id == id)
            .ok_or_else(|| "下载任务不存在".to_string())?;
        if task.status.terminal() {
            return Err("下载任务已经结束".into());
        }
        if let Some(flag) = cancelled {
            flag.store(true, Ordering::Relaxed);
        }
        task.status = DownloadStatus::Cancelled;
        task.message = Some("已取消".into());
        task.updated_at = now_ms();
        Ok(task.clone())
    }

    fn mark_cancelled(&self, id: &str) -> Option<DownloadTask> {
        let mut inner = self.inner.lock().ok()?;
        let task = inner.tasks.iter_mut().find(|task| task.id == id)?;
        if !task.status.terminal() {
            task.status = DownloadStatus::Cancelled;
            task.message = Some("已取消".into());
            task.updated_at = now_ms();
        }
        Some(task.clone())
    }

    fn remove(&self, id: &str) -> Result<Vec<DownloadTask>, String> {
        let mut inner = self.inner.lock().map_err(|_| "下载状态锁已损坏")?;
        let task = inner
            .tasks
            .iter()
            .find(|task| task.id == id)
            .ok_or_else(|| "下载任务不存在".to_string())?;
        if !task.status.terminal() {
            return Err("只能移除已结束的下载任务".into());
        }
        inner.controls.remove(id);
        inner.tasks.retain(|task| task.id != id);
        Ok(inner.tasks.iter().rev().cloned().collect())
    }

    fn clear_finished(&self) -> Vec<DownloadTask> {
        let Ok(mut inner) = self.inner.lock() else {
            return Vec::new();
        };
        let finished = inner
            .tasks
            .iter()
            .filter(|task| task.status.terminal())
            .map(|task| task.id.clone())
            .collect::<Vec<_>>();
        for id in finished {
            inner.controls.remove(&id);
        }
        inner.tasks.retain(|task| !task.status.terminal());
        inner.tasks.iter().rev().cloned().collect()
    }

    fn is_cancelled(&self, id: &str) -> bool {
        self.inner
            .lock()
            .ok()
            .and_then(|inner| {
                inner
                    .controls
                    .get(id)
                    .map(|flag| flag.load(Ordering::Relaxed))
            })
            .unwrap_or(false)
    }

    async fn acquire(&self) -> Result<OwnedSemaphorePermit, String> {
        self.slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| "下载队列已关闭".to_string())
    }
}

#[tauri::command]
pub(crate) async fn start_download(
    app: AppHandle,
    manager: tauri::State<'_, DownloadManager>,
    repo: String,
    source: Source,
    variant: Option<String>,
) -> Result<DownloadTask, String> {
    let manager = manager.inner().clone();
    let task = manager.enqueue(repo.trim().to_string(), source, variant)?;
    emit_task(&app, &task);
    let app = app.clone();
    let id = task.id.clone();
    tauri::async_runtime::spawn(async move {
        run_download(app, manager, id).await;
    });
    Ok(task)
}

#[tauri::command]
pub(crate) async fn start_backend_download(
    app: AppHandle,
    manager: tauri::State<'_, DownloadManager>,
    package: BackendManifest,
) -> Result<DownloadTask, String> {
    let manager = manager.inner().clone();
    let task = manager.enqueue_backend(package)?;
    emit_task(&app, &task);
    let app = app.clone();
    let id = task.id.clone();
    tauri::async_runtime::spawn(async move {
        run_download(app, manager, id).await;
    });
    Ok(task)
}

#[tauri::command]
pub(crate) fn list_downloads(manager: tauri::State<'_, DownloadManager>) -> Vec<DownloadTask> {
    manager.list()
}

#[tauri::command]
pub(crate) fn cancel_download(
    app: AppHandle,
    manager: tauri::State<'_, DownloadManager>,
    id: String,
) -> Result<DownloadTask, String> {
    let task = manager.cancel(&id)?;
    emit_task(&app, &task);
    Ok(task)
}

#[tauri::command]
pub(crate) fn remove_download(
    manager: tauri::State<'_, DownloadManager>,
    id: String,
) -> Result<Vec<DownloadTask>, String> {
    manager.remove(&id)
}

#[tauri::command]
pub(crate) fn clear_downloads(manager: tauri::State<'_, DownloadManager>) -> Vec<DownloadTask> {
    manager.clear_finished()
}

async fn run_download(app: AppHandle, manager: DownloadManager, id: String) {
    let permit = match manager.acquire().await {
        Ok(permit) => permit,
        Err(error) => {
            set_status(&app, &manager, &id, DownloadStatus::Failed, Some(error));
            return;
        }
    };
    if manager.is_cancelled(&id) {
        if let Some(task) = manager.mark_cancelled(&id) {
            emit_task(&app, &task);
        }
        return;
    }
    set_status(
        &app,
        &manager,
        &id,
        DownloadStatus::Preparing,
        Some("读取仓库文件…".into()),
    );
    let Some(task) = manager.get(&id) else {
        return;
    };
    if task.kind == DownloadKind::Backend {
        run_backend_download(&app, &manager, &task).await;
    } else {
        let result = install_model(&app, &manager, &task).await;
        match result {
            Ok((total, target)) => {
                if let Some(task) = manager.update(&id, |task| {
                    task.total = total;
                    task.downloaded = total;
                    task.path = Some(target.display().to_string());
                    task.file = None;
                    task.status = DownloadStatus::Done;
                    task.message = Some("下载完成并已安装".into());
                }) {
                    emit_task(&app, &task);
                }
            }
            Err(InstallError::Cancelled) => {
                if let Some(task) = manager.mark_cancelled(&id) {
                    emit_task(&app, &task);
                }
            }
            Err(InstallError::Message(message)) => {
                if let Some(task) = manager.update(&id, |task| {
                    task.status = DownloadStatus::Failed;
                    task.message = Some(message);
                }) {
                    emit_task(&app, &task);
                }
            }
        }
    }
    drop(permit);
}

async fn run_backend_download(app: &AppHandle, manager: &DownloadManager, task: &DownloadTask) {
    let Some(package) = task.backend.clone() else {
        if let Some(task) = manager.update(&task.id, |task| {
            task.status = DownloadStatus::Failed;
            task.message = Some("后端包描述缺失".into());
        }) {
            emit_task(app, &task);
        }
        return;
    };
    set_status(
        app,
        manager,
        &task.id,
        DownloadStatus::Preparing,
        Some("准备下载引擎包…".into()),
    );
    let total = task.total;
    let manager_for_progress = manager.clone();
    let app_for_progress = app.clone();
    let id_for_progress = task.id.clone();
    let progress = move |downloaded: u64, total: u64| {
        let status = if total > 0 && downloaded >= total {
            DownloadStatus::Verifying
        } else {
            DownloadStatus::Downloading
        };
        if let Some(task) = manager_for_progress.update(&id_for_progress, |task| {
            task.downloaded = downloaded;
            task.total = total;
            task.status = status;
            task.message =
                (status == DownloadStatus::Verifying).then(|| "校验并解压引擎包…".into());
        }) {
            emit_task(&app_for_progress, &task);
        }
    };
    let manager_for_cancel = manager.clone();
    let id_for_cancel = task.id.clone();
    let cancel = move || manager_for_cancel.is_cancelled(&id_for_cancel);
    let result = crate::backend_store::install_package_with(package, progress, cancel).await;
    match result {
        Ok(installed) => {
            if let Some(task) = manager.update(&task.id, |task| {
                task.downloaded = total;
                task.total = total;
                task.path = Some(installed.path.display().to_string());
                task.status = DownloadStatus::Done;
                task.message = Some("下载完成并已安装".into());
            }) {
                emit_task(app, &task);
            }
        }
        Err(error) if error == crate::backend_store::download::CANCELLED => {
            if let Some(task) = manager.mark_cancelled(&task.id) {
                emit_task(app, &task);
            }
        }
        Err(error) => {
            if let Some(task) = manager.update(&task.id, |task| {
                task.status = DownloadStatus::Failed;
                task.message = Some(error);
            }) {
                emit_task(app, &task);
            }
        }
    }
}

fn emit_progress(
    app: Option<&AppHandle>,
    manager: &DownloadManager,
    id: &str,
    file: &RemoteFile,
    downloaded: u64,
    total: u64,
) {
    if let Some(task) = manager.update(id, |task| {
        task.file = Some(file.path.clone());
        task.downloaded = downloaded;
        task.total = total;
        task.status = DownloadStatus::Downloading;
    }) {
        if let Some(app) = app {
            emit_task(app, &task);
        }
    }
}

fn set_status(
    app: &AppHandle,
    manager: &DownloadManager,
    id: &str,
    status: DownloadStatus,
    message: Option<String>,
) {
    if let Some(task) = manager.update(id, |task| {
        task.status = status;
        task.message = message;
    }) {
        emit_task(app, &task);
    }
}

fn emit_task(app: &AppHandle, task: &DownloadTask) {
    let _ = app.emit("model-download", task.clone());
}

fn cleanup_stale_staging() {
    let root = models_root();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    let cutoff = SystemTime::now() - Duration::from_secs(60 * 60);
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with('.') || !name.contains(".installing-") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() && metadata.modified().is_ok_and(|modified| modified < cutoff) {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

static NEXT_TASK: AtomicU64 = AtomicU64::new(1);

fn next_task_id() -> String {
    let sequence = NEXT_TASK.fetch_add(1, Ordering::Relaxed);
    format!("download-{}-{sequence}", now_ms())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_rejects_duplicates_and_removes_finished_tasks() {
        let manager = DownloadManager::new();
        let task = manager
            .enqueue("Qwen/Qwen3-0.6B-GGUF".into(), Source::ModelScope, None)
            .expect("first task");
        assert!(
            manager
                .enqueue("Qwen/Qwen3-0.6B-GGUF".into(), Source::ModelScope, None)
                .is_err()
        );
        assert!(manager.remove(&task.id).is_err());
        let cancelled = manager.cancel(&task.id).expect("cancel");
        assert_eq!(cancelled.status, DownloadStatus::Cancelled);
        assert!(manager.remove(&task.id).expect("remove").is_empty());
        assert!(manager.list().is_empty());
    }

    fn backend_package() -> BackendManifest {
        BackendManifest {
            id: "llama-vulkan".into(),
            name: "llama.cpp".into(),
            version: "b1".into(),
            description: String::new(),
            homepage: None,
            license: None,
            platform: "windows-x86_64".into(),
            worker: "llama-server.exe".into(),
            launcher: uniinfer_backend_store::BackendLauncher::LlamaServer,
            devices: vec![uniinfer_core::DeviceClass::Gpu],
            engines: vec![uniinfer_backend_store::BackendEngine {
                id: "llama-vulkan".into(),
                name: "llama.cpp".into(),
                runtime: "llama".into(),
                description: String::new(),
                vendor: "llama.cpp".into(),
                variant: "Vulkan".into(),
                category: "language".into(),
                homepage: None,
                implementations: vec!["llama.text".into()],
                formats: vec!["gguf".into()],
                modalities: vec!["text".into()],
                devices: vec![uniinfer_core::DeviceClass::Gpu],
            }],
            files: vec![uniinfer_backend_store::BackendFile {
                path: "llama.zip".into(),
                url: "https://example.com/llama.zip".into(),
                sha256: None,
                size: 42,
            }],
        }
    }

    #[test]
    fn backend_tasks_share_the_model_queue() {
        let manager = DownloadManager::new();
        let task = manager
            .enqueue_backend(backend_package())
            .expect("backend task");
        assert_eq!(task.kind, DownloadKind::Backend);
        assert_eq!(task.source, Source::Backend);
        assert_eq!(task.variant_label.as_deref(), Some("Vulkan"));
        assert_eq!(task.total, 42);
        assert!(manager.enqueue_backend(backend_package()).is_err());
        manager.cancel(&task.id).expect("cancel");
        assert!(manager.remove(&task.id).expect("remove").is_empty());
    }

    #[test]
    fn terminal_statuses_are_distinct_from_active_ones() {
        assert!(!DownloadStatus::Queued.terminal());
        assert!(!DownloadStatus::Downloading.terminal());
        assert!(DownloadStatus::Done.terminal());
        assert!(DownloadStatus::Failed.terminal());
        assert!(DownloadStatus::Cancelled.terminal());
    }
}
