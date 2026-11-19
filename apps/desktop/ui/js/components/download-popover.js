import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, errorMessage } from '../format.js';
import { toast } from './ui.js';
import {
  cancelDownload,
  clearFinishedDownloads,
  initDownloads,
  isTerminalStatus,
  pauseDownload,
  removeDownload,
  resumeDownload,
  subscribeDownloads,
} from '../downloads.js';

const statusLabels = {
  queued: '排队中',
  preparing: '准备中',
  downloading: '下载中',
  paused: '已暂停',
  verifying: '校验中',
  inspecting: '生成配置',
  committing: '安装中',
  done: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

function sourceLabel(source) {
  if (source === 'backend') return '推理后端';
  return source === 'huggingface' ? 'Hugging Face' : 'ModelScope';
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—';
  if (seconds < 60) return `${Math.ceil(seconds)} 秒`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} 分钟`;
  return `${(seconds / 3600).toFixed(1)} 小时`;
}

export function createDownloadPopover() {
  let open = false;
  let tasks = [];
  let renderScheduled = false;
  const samples = new Map();

  const badge = h('span', { class: 'download-badge', hidden: true, text: '' });
  const button = h('button', {
    class: 'btn btn-icon ghost download-toggle', type: 'button', title: '下载任务',
    attrs: { 'aria-label': '下载任务', 'aria-expanded': 'false' },
    onClick: (event) => {
      event.stopPropagation();
      toggle();
    },
  }, h('span', { html: icon('download', 16) }), badge);
  const summary = h('span', { class: 'download-popover-summary faint' });
  const clearButton = h('button', {
    class: 'btn sm ghost', type: 'button', text: '清除已结束',
    onClick: () => void clearFinished(),
  });
  const list = h('div', { class: 'download-popover-list' });
  const panel = h('section', { class: 'download-popover', hidden: true },
    h('div', { class: 'download-popover-head' },
      h('div', { class: 'download-popover-title' },
        h('span', { html: icon('download', 15) }),
        h('strong', { text: '下载任务' }),
        summary,
      ),
      clearButton,
    ),
    list,
  );
  const root = h('div', { class: 'download-popover-root' }, button, panel);

  function setOpen(next) {
    open = next;
    panel.hidden = !open;
    button.classList.toggle('active', open);
    button.setAttribute('aria-expanded', String(open));
    if (open) render();
  }

  function toggle() {
    setOpen(!open);
  }

  function close() {
    if (open) setOpen(false);
  }

  function activeCount() {
    return tasks.filter((task) => !isTerminalStatus(task.status)).length;
  }

  function updateBadge() {
    const count = activeCount();
    badge.hidden = count === 0;
    badge.textContent = count > 99 ? '99+' : String(count);
  }

  function scheduleRender() {
    if (renderScheduled) return;
    renderScheduled = true;
    requestAnimationFrame(() => {
      renderScheduled = false;
      render();
    });
  }

  function render() {
    updateBadge();
    if (!open) return;
    summary.textContent = `${activeCount()} 个进行中 · ${tasks.length} 个任务`;
    clearButton.disabled = !tasks.some((task) => isTerminalStatus(task.status));
    clear(list);
    if (!tasks.length) {
      list.append(h('div', { class: 'download-popover-empty' },
        h('div', { class: 'download-popover-empty-icon', html: icon('download', 20) }),
        h('strong', { text: '还没有下载任务' }),
        h('span', { text: '从模型或后端页面加入下载后，会出现在这里。' }),
      ));
      return;
    }
    for (const task of tasks) list.append(taskCard(task));
  }

  function taskCard(task) {
    const terminal = isTerminalStatus(task.status);
    const total = Number(task.total) || 0;
    const downloaded = Number(task.downloaded) || 0;
    const percent = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : 0;
    const speed = !terminal && task.status === 'downloading' ? updateSpeed(task) : 0;
    const eta = speed > 0 && total > downloaded ? (total - downloaded) / speed : 0;
    const queued = task.status === 'queued';
    const bar = h('div', {
      class: `progress ${queued ? 'queued' : total > 0 ? 'determinate' : ''}`,
    }, h('span'));
    if (!queued && total > 0) bar.firstChild.style.width = `${percent}%`;
    if (task.status === 'failed' || task.status === 'cancelled') bar.hidden = true;
    const meta = [
      task.kind === 'backend' ? null : sourceLabel(task.source),
      task.variant_label || task.variant,
      task.file ? task.file.split('/').pop() : null,
    ].filter(Boolean);
    const kind = task.status === 'done' ? 'success'
      : task.status === 'failed' ? 'error'
        : task.status === 'cancelled' ? 'muted'
          : task.status === 'paused' ? 'warning' : 'accent';
    const actions = h('div', { class: 'download-popover-actions' });
    if (terminal) {
      if (task.status !== 'done') {
        actions.append(h('button', {
          class: 'btn sm ghost', type: 'button', text: '继续',
          onClick: () => void resume(task),
        }));
      }
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '移除',
        onClick: () => void remove(task),
      }));
    } else if (task.status === 'paused') {
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '继续',
        onClick: () => void resume(task),
      }));
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '取消',
        onClick: () => void cancel(task),
      }));
    } else {
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '暂停',
        onClick: () => void pause(task),
      }));
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '取消',
        onClick: () => void cancel(task),
      }));
    }
    return h('article', { class: 'download-popover-task', dataset: { task: task.id } },
      h('div', { class: 'download-popover-task-head' },
        h('span', { class: 'download-popover-task-icon' }, h('span', { html: icon(task.status === 'done' ? 'check' : 'download', 14) })),
        h('div', { class: 'download-popover-task-title' },
          h('strong', { text: task.repo }),
          h('span', { text: meta.join(' · ') || task.model_id }),
        ),
        h('span', { class: `chip ${kind}`, text: statusLabels[task.status] ?? task.status }),
      ),
      bar,
      h('div', { class: 'download-popover-stats faint' },
        h('span', { text: total > 0 ? `${formatBytes(downloaded)} / ${formatBytes(total)} · ${percent}%` : `${formatBytes(downloaded)} · 大小未知` }),
        speed > 0 ? h('span', { text: `${formatBytes(speed)}/s` }) : null,
        eta > 0 ? h('span', { text: `剩余 ${formatEta(eta)}` }) : null,
      ),
      task.message && task.status !== 'downloading'
        ? h('div', { class: `download-popover-message ${task.status === 'failed' ? 'error' : 'faint'}`, text: task.message })
        : null,
      actions,
    );
  }

  function updateSpeed(task) {
    const now = Date.now();
    const previous = samples.get(task.id);
    if (!previous || task.downloaded < previous.bytes || now <= previous.at) {
      samples.set(task.id, { at: now, bytes: task.downloaded, speed: 0 });
      return previous?.speed ?? 0;
    }
    const elapsed = (now - previous.at) / 1000;
    if (elapsed < 0.4) return previous.speed;
    const speed = (task.downloaded - previous.bytes) / elapsed;
    samples.set(task.id, { at: now, bytes: task.downloaded, speed });
    return speed;
  }

  async function clearFinished() {
    try {
      tasks = await clearFinishedDownloads();
      render();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function pause(task) {
    try {
      await pauseDownload(task.id);
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function resume(task) {
    try {
      await resumeDownload(task.id);
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function cancel(task) {
    try {
      await cancelDownload(task.id);
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function remove(task) {
    try {
      tasks = await removeDownload(task.id);
      render();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }


  const unsubscribe = subscribeDownloads((next) => {
    tasks = next;
    scheduleRender();
  });
  document.addEventListener('click', (event) => {
    if (open && root.isConnected && !root.contains(event.target)) close();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') close();
  });
  void initDownloads().then((next) => {
    tasks = next;
    scheduleRender();
  }).catch(() => {});

  return {
    element: root,
    open() {
      setOpen(true);
    },
    close,
    dispose() {
      unsubscribe();
    },
  };
}
