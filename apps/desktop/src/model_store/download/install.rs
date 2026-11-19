//! Install orchestration: fetch plan, staging, inspect and atomic commit.

use std::path::PathBuf;
use std::time::Duration;

use tauri::AppHandle;
use uniinfer_model_inspect::{InspectOptions, inspect};

use super::super::plan::{build_variants, choose_variant};
use super::super::repo::fetch_repo_files;
use super::super::{commit_install, models_root};
use super::fetch::{InstallError, check_cancelled, download_file, safe_join};
use super::{DownloadManager, DownloadStatus, DownloadTask, emit_task, set_status};

pub(super) async fn install_model(
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
    let staging = task
        .staging
        .as_deref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| models_root().join(format!(".{}.installing", task.model_id)));
    tokio::fs::create_dir_all(&staging)
        .await
        .map_err(|error| InstallError::Message(format!("create {}: {error}", staging.display())))?;
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
