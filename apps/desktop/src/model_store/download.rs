//! Download queue, progress, cancellation and resumable HTTP downloads.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use reqwest::StatusCode;
use reqwest::header::{ACCEPT_ENCODING, RANGE};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_model_inspect::{InspectOptions, inspect};

use super::plan::{build_variants, choose_variant};
use super::repo::fetch_repo_files;
use super::{
    RemoteFile, Source, StagingCleanup, commit_install, models_root, slug, temp_model_path,
};

const PARALLEL_DOWNLOADS: usize = 1;
const DOWNLOAD_ATTEMPTS: usize = 3;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(120);
const READ_POLL: Duration = Duration::from_millis(500);
const READ_IDLE: Duration = Duration::from_secs(60);

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

#[derive(Clone, Debug, Serialize)]
pub(crate) struct DownloadTask {
    pub(crate) id: String,
    pub(crate) repo: String,
    pub(crate) source: Source,
    pub(crate) variant: Option<String>,
    pub(crate) variant_label: Option<String>,
    pub(crate) model_id: String,
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
            repo,
            source,
            variant,
            variant_label: None,
            model_id,
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
    drop(permit);
}

async fn install_model(
    app: &AppHandle,
    manager: &DownloadManager,
    task: &DownloadTask,
) -> Result<(u64, PathBuf), InstallError> {
    let (repo_files, base_files) = fetch_repo_files(&task.repo, task.source)
        .await
        .map_err(InstallError::Message)?;
    check_cancelled(manager, &task.id)?;
    let variants =
        build_variants(&repo_files, base_files.as_ref()).map_err(InstallError::Message)?;
    let chosen =
        choose_variant(&variants, task.variant.as_deref()).map_err(InstallError::Message)?;
    let total = chosen.files.iter().map(|file| file.size).sum::<u64>();
    let target = models_root().join(&task.model_id);
    let staging = temp_model_path(&models_root(), &task.model_id, "installing");
    tokio::fs::create_dir_all(&staging)
        .await
        .map_err(|error| InstallError::Message(format!("create {}: {error}", staging.display())))?;
    let _cleanup = StagingCleanup(staging.clone());
    if let Some(task) = manager.update(&task.id, |task| {
        task.variant_label = Some(chosen.label.clone());
        task.total = total;
        task.status = DownloadStatus::Downloading;
        task.message = None;
    }) {
        emit_task(app, &task);
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .user_agent("UniInfer-Desktop/0.0")
        .build()
        .map_err(|error| InstallError::Message(format!("http client: {error}")))?;
    let mut done = 0u64;
    for file in &chosen.files {
        check_cancelled(manager, &task.id)?;
        let destination = safe_join(&staging, &file.path)?;
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                InstallError::Message(format!("create {}: {error}", parent.display()))
            })?;
        }
        if let Some(update) = manager.update(&task.id, |task| {
            task.file = Some(file.path.clone());
            task.downloaded = done;
        }) {
            emit_task(app, &update);
        }
        done += download_file(Some(app), manager, &task.id, &client, file, &destination).await?;
        if let Some(update) = manager.update(&task.id, |task| {
            task.downloaded = done;
        }) {
            emit_task(app, &update);
        }
    }

    set_status(
        app,
        manager,
        &task.id,
        DownloadStatus::Verifying,
        Some("校验文件…".into()),
    );
    check_cancelled(manager, &task.id)?;
    set_status(
        app,
        manager,
        &task.id,
        DownloadStatus::Inspecting,
        Some("生成模型配置…".into()),
    );
    let directory = staging.clone();
    let model_id = task.model_id.clone();
    tokio::task::spawn_blocking(move || {
        let inspection = inspect(
            &directory,
            &InspectOptions {
                id: Some(model_id),
                ..InspectOptions::default()
            },
        )
        .map_err(|error| format!("无法为下载的模型生成 manifest: {error}"))?;
        inspection
            .write_manifest()
            .map_err(|error| format!("写入 manifest: {error}"))
    })
    .await
    .map_err(|error| InstallError::Message(format!("inspect task: {error}")))?
    .map_err(InstallError::Message)?;

    check_cancelled(manager, &task.id)?;
    set_status(
        app,
        manager,
        &task.id,
        DownloadStatus::Committing,
        Some("安装到模型库…".into()),
    );
    let commit_staging = staging.clone();
    let commit_target = target.clone();
    tokio::task::spawn_blocking(move || commit_install(&commit_staging, &commit_target))
        .await
        .map_err(|error| InstallError::Message(format!("install commit task: {error}")))?
        .map_err(InstallError::Message)?;
    Ok((total.max(done), target))
}

async fn download_file(
    app: Option<&AppHandle>,
    manager: &DownloadManager,
    task_id: &str,
    client: &reqwest::Client,
    file: &RemoteFile,
    destination: &Path,
) -> Result<u64, InstallError> {
    let done_before = manager
        .get(task_id)
        .map(|task| task.downloaded)
        .unwrap_or(0);
    let total = manager.get(task_id).map(|task| task.total).unwrap_or(0);
    let part = part_path(destination);
    if destination.is_file() {
        let length = tokio::fs::metadata(destination)
            .await
            .map_err(|error| {
                InstallError::Message(format!("stat {}: {error}", destination.display()))
            })?
            .len();
        if file.size == 0 || length == file.size {
            return Ok(length);
        }
    }
    let mut current = match tokio::fs::metadata(&part).await {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => {
            return Err(InstallError::Message(format!(
                "stat {}: {error}",
                part.display()
            )));
        }
    };
    if file.size > 0 && current > file.size {
        tokio::fs::remove_file(&part).await.map_err(|error| {
            InstallError::Message(format!("remove {}: {error}", part.display()))
        })?;
        current = 0;
    }
    if file.size > 0 && current == file.size && verify_part(&part, file).await? {
        rename_part(&part, destination).await?;
        emit_progress(app, manager, task_id, file, done_before + current, total);
        return Ok(current);
    }
    if file.size > 0 && current == file.size {
        let _ = tokio::fs::remove_file(&part).await;
        current = 0;
    }

    let mut last_error = String::new();
    for attempt in 0..DOWNLOAD_ATTEMPTS {
        check_cancelled(manager, task_id)?;
        let mut request = client.get(&file.url).header(ACCEPT_ENCODING, "identity");
        if current > 0 {
            request = request.header(RANGE, format!("bytes={current}-"));
        }
        let response = match tokio::time::timeout(Duration::from_secs(60), request.send()).await {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                last_error = format!("download {}: {error}", file.url);
                retry_or_break(attempt, &mut last_error).await?;
                continue;
            }
            Err(_) => {
                last_error = format!("download {}: 请求超时", file.url);
                retry_or_break(attempt, &mut last_error).await?;
                continue;
            }
        };
        let status = response.status();
        let resumed = current > 0 && status == StatusCode::PARTIAL_CONTENT;
        if !resumed && status != StatusCode::OK {
            last_error = format!("download {}: HTTP {status}", file.url);
            if status.is_client_error() && status != StatusCode::REQUEST_TIMEOUT {
                break;
            }
            retry_or_break(attempt, &mut last_error).await?;
            continue;
        }
        if !resumed {
            current = 0;
        }
        let mut output = if resumed {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&part)
                .await
                .map_err(|error| {
                    InstallError::Message(format!("open {}: {error}", part.display()))
                })?
        } else {
            tokio::fs::File::create(&part).await.map_err(|error| {
                InstallError::Message(format!("create {}: {error}", part.display()))
            })?
        };
        let mut hasher = if file.sha256.is_some() {
            Some(if resumed {
                hash_existing(&part).await?
            } else {
                Sha256::new()
            })
        } else {
            None
        };
        let mut stream = response.bytes_stream();
        let mut last_emit = Instant::now();
        let mut last_data = Instant::now();
        let mut stream_error = None;
        emit_progress(app, manager, task_id, file, done_before + current, total);
        loop {
            check_cancelled(manager, task_id)?;
            let next = tokio::time::timeout(READ_POLL, stream.next()).await;
            match next {
                Err(_) => {
                    if last_data.elapsed() >= READ_IDLE {
                        stream_error = Some(format!("download {}: 读取超时", file.url));
                        break;
                    }
                    continue;
                }
                Ok(Some(Ok(chunk))) => {
                    last_data = Instant::now();
                    current += chunk.len() as u64;
                    if let Some(hasher) = &mut hasher {
                        hasher.update(&chunk);
                    }
                    output.write_all(&chunk).await.map_err(|error| {
                        InstallError::Message(format!("write {}: {error}", part.display()))
                    })?;
                    if last_emit.elapsed() >= PROGRESS_INTERVAL {
                        emit_progress(app, manager, task_id, file, done_before + current, total);
                        last_emit = Instant::now();
                    }
                }
                Ok(Some(Err(error))) => {
                    stream_error = Some(format!("download {}: {error}", file.url));
                    break;
                }
                Ok(None) => break,
            }
        }
        output
            .flush()
            .await
            .map_err(|error| InstallError::Message(format!("flush {}: {error}", part.display())))?;
        drop(output);
        if let Some(error) = stream_error {
            last_error = error;
            retry_or_break(attempt, &mut last_error).await?;
            continue;
        }
        if file.size > 0 && current != file.size {
            last_error = format!(
                "download {}: expected {} bytes, got {current}",
                file.url, file.size
            );
            retry_or_break(attempt, &mut last_error).await?;
            continue;
        }
        if let Some(expected) = &file.sha256 {
            let actual = hex::encode(hasher.take().unwrap_or_default().finalize());
            if !actual.eq_ignore_ascii_case(expected) {
                let _ = tokio::fs::remove_file(&part).await;
                current = 0;
                last_error = format!(
                    "checksum mismatch for {}: expected {expected}, got {actual}",
                    file.path
                );
                retry_or_break(attempt, &mut last_error).await?;
                continue;
            }
        }
        rename_part(&part, destination).await?;
        emit_progress(app, manager, task_id, file, done_before + current, total);
        return Ok(current);
    }
    Err(InstallError::Message(last_error))
}

async fn retry_or_break(attempt: usize, last_error: &mut String) -> Result<(), InstallError> {
    if attempt + 1 >= DOWNLOAD_ATTEMPTS {
        return Err(InstallError::Message(std::mem::take(last_error)));
    }
    tokio::time::sleep(Duration::from_millis(400 * (attempt as u64 + 1))).await;
    Ok(())
}

async fn verify_part(path: &Path, file: &RemoteFile) -> Result<bool, InstallError> {
    let Some(expected) = &file.sha256 else {
        return Ok(true);
    };
    let actual = hex::encode(hash_existing(path).await?.finalize());
    Ok(actual.eq_ignore_ascii_case(expected))
}

async fn hash_existing(path: &Path) -> Result<Sha256, InstallError> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|error| InstallError::Message(format!("open {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(|error| InstallError::Message(format!("read {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher)
}

async fn rename_part(part: &Path, destination: &Path) -> Result<(), InstallError> {
    tokio::fs::rename(part, destination).await.map_err(|error| {
        InstallError::Message(format!(
            "rename {} -> {}: {error}",
            part.display(),
            destination.display()
        ))
    })
}

fn part_path(destination: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "download".into());
    destination.with_file_name(format!("{name}.part"))
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, InstallError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(InstallError::Message(format!(
            "模型文件路径不安全：{relative}"
        )));
    }
    Ok(root.join(path))
}

fn check_cancelled(manager: &DownloadManager, id: &str) -> Result<(), InstallError> {
    if manager.is_cancelled(id) {
        Err(InstallError::Cancelled)
    } else {
        Ok(())
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

#[derive(Debug)]
enum InstallError {
    Cancelled,
    Message(String),
}

impl From<String> for InstallError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
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
    fn rejects_paths_that_escape_the_staging_directory() {
        let root = Path::new("staging");
        assert!(safe_join(root, "model.gguf").is_ok());
        assert!(safe_join(root, "sub/model.gguf").is_ok());
        assert!(safe_join(root, "../model.gguf").is_err());
        assert!(safe_join(root, "/tmp/model.gguf").is_err());
        assert!(safe_join(root, "sub/../model.gguf").is_err());
    }

    #[test]
    fn keeps_part_files_next_to_their_destination() {
        let part = part_path(Path::new("models/model.gguf"));
        assert_eq!(part, PathBuf::from("models/model.gguf.part"));
    }

    #[tokio::test]
    async fn resumes_partial_download_and_verifies_checksum() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let payload = b"UniInfer downloadable payload".repeat(4096);
        let total = payload.len();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let range_seen = Arc::new(AtomicBool::new(false));
        let range_flag = range_seen.clone();
        let body = payload.clone();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept");
            let mut request = [0u8; 4096];
            let read = socket.read(&mut request).expect("read request");
            let request = String::from_utf8_lossy(&request[..read]);
            let start = request
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("range: bytes=")
                        .and_then(|value| value.split('-').next())
                        .and_then(|value| value.parse::<usize>().ok())
                })
                .unwrap_or(0);
            if start > 0 {
                range_flag.store(true, Ordering::Relaxed);
            }
            let body = &body[start..];
            let header = if start > 0 {
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    body.len(),
                    start,
                    total - 1,
                    total
                )
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
            };
            socket.write_all(header.as_bytes()).expect("headers");
            socket.write_all(body).expect("body");
        });

        let directory = tempfile::tempdir().expect("tempdir");
        let destination = directory.path().join("artifact.bin");
        let partial = part_path(&destination);
        tokio::fs::write(&partial, &payload[..17])
            .await
            .expect("partial");
        let manager = DownloadManager::new();
        let task = manager
            .enqueue("test/model".into(), Source::HuggingFace, None)
            .expect("task");
        let file = RemoteFile {
            path: "artifact.bin".into(),
            url: format!("http://{address}/artifact.bin"),
            sha256: Some(hex::encode(Sha256::digest(&payload))),
            size: payload.len() as u64,
        };
        let client = reqwest::Client::new();
        let downloaded = download_file(None, &manager, &task.id, &client, &file, &destination)
            .await
            .expect("download");
        assert_eq!(downloaded, payload.len() as u64);
        assert_eq!(tokio::fs::read(&destination).await.expect("file"), payload);
        assert!(!partial.exists());
        assert!(range_seen.load(Ordering::Relaxed));
        server.join().expect("server");
    }

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

    #[test]
    fn terminal_statuses_are_distinct_from_active_ones() {
        assert!(!DownloadStatus::Queued.terminal());
        assert!(!DownloadStatus::Downloading.terminal());
        assert!(DownloadStatus::Done.terminal());
        assert!(DownloadStatus::Failed.terminal());
        assert!(DownloadStatus::Cancelled.terminal());
    }
}
