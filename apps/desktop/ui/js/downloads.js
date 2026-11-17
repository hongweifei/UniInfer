const invoke = (command, args) => window.__TAURI__?.core?.invoke(command, args);

const terminalStatuses = new Set(['done', 'failed', 'cancelled']);
const listeners = new Set();
let tasks = [];
let initialized = false;
let unlisten = null;

export function subscribeDownloads(listener) {
  listeners.add(listener);
  listener(tasks);
  return () => listeners.delete(listener);
}

export function downloadTasks() {
  return tasks;
}

export function activeDownloads() {
  return tasks.filter((task) => !terminalStatuses.has(task.status));
}

export function findDownload(repo, source) {
  return tasks.find((task) => task.repo === repo && task.source === source) ?? null;
}

export function taskById(id) {
  return tasks.find((task) => task.id === id) ?? null;
}

export function isTerminalStatus(status) {
  return terminalStatuses.has(status);
}

export async function initDownloads() {
  if (initialized) return tasks;
  initialized = true;
  if (!window.__TAURI__?.core?.invoke) return tasks;
  try {
    tasks = (await invoke('list_downloads')) ?? [];
    if (window.__TAURI__?.event?.listen) {
      unlisten = await window.__TAURI__.event.listen('model-download', (event) => {
        upsertTask(event.payload);
      });
    }
  } catch {
    // Downloads are optional; the catalog will surface a concrete error when starting one.
  }
  notify();
  return tasks;
}

export async function startDownload({ repo, source, variant }) {
  const task = await invoke('start_download', { repo, source, variant: variant ?? null });
  upsertTask(task);
  return task;
}

export async function startBackendDownload(packageInfo) {
  const task = await invoke('start_backend_download', { package: packageInfo });
  upsertTask(task);
  return task;
}

export async function cancelDownload(id) {
  const task = await invoke('cancel_download', { id });
  upsertTask(task);
  return task;
}

export async function removeDownload(id) {
  tasks = (await invoke('remove_download', { id })) ?? [];
  notify();
  return tasks;
}

export async function clearFinishedDownloads() {
  tasks = (await invoke('clear_downloads')) ?? [];
  notify();
  return tasks;
}

export function disposeDownloads() {
  if (unlisten) unlisten();
  unlisten = null;
}

function upsertTask(task) {
  if (!task?.id) return;
  const index = tasks.findIndex((item) => item.id === task.id);
  if (index >= 0) {
    tasks = tasks.map((item) => (item.id === task.id ? { ...item, ...task } : item));
  } else {
    tasks = [task, ...tasks];
  }
  notify();
}

function notify() {
  for (const listener of [...listeners]) listener(tasks);
}
