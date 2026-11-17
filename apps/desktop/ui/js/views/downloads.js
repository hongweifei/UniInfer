import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { errorMessage, formatBytes } from '../format.js';
import { toast, emptyState } from '../components/ui.js';
import {
  cancelDownload,
  clearFinishedDownloads,
  initDownloads,
  isTerminalStatus,
  removeDownload,
  startDownload,
  startBackendDownload,
  subscribeDownloads,
} from '../downloads.js';

const statusLabels = {
  queued: '排队中',
  preparing: '准备中',
  downloading: '下载中',
  verifying: '校验中',
  inspecting: '生成配置',
  committing: '安装中',
  done: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

export function mountDownloads(root, ctx) {
  const summary = h('div', { class: 'row download-summary' });
  const clearButton = h('button', {
    class: 'btn sm ghost', type: 'button', onClick: () => void clearFinished(),
  }, h('span', { html: icon('trash', 13) }), h('span', { text: '清除已结束' }));
  const list = h('div', { class: 'download-list' });
  const samples = new Map();

  function clearFinished() {
    clearButton.disabled = true;
    void clearFinishedDownloads()
      .catch((error) => toast(errorMessage(error), 'error'))
      .finally(() => render(currentTasks));
  }

  let currentTasks = [];
  function render(tasks) {
    currentTasks = tasks;
    const active = tasks.filter((task) => !isTerminalStatus(task.status));
    summary.replaceChildren(
      h('span', { class: 'faint', text: active.length ? `${active.length} 个任务进行中` : '没有进行中的下载' }),
      h('span', { class: 'spacer' }),
      clearButton,
    );
    clearButton.disabled = !tasks.some((task) => isTerminalStatus(task.status));
    clear(list);
    if (!tasks.length) {
      list.append(emptyState({
        iconName: 'download',
        title: '还没有下载任务',
        description: '在线模型和推理后端共用这个队列，按顺序下载并显示进度。',
      }));
      return;
    }
    for (const task of tasks) list.append(taskCard(task));
  }

  function taskCard(task) {
    const active = !isTerminalStatus(task.status);
    const total = Number(task.total) || 0;
    const downloaded = Number(task.downloaded) || 0;
    const percent = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : 0;
    const bar = h('div', { class: `progress ${total > 0 ? 'determinate' : ''}` }, h('span'));
    if (total > 0) bar.firstChild.style.width = `${percent}%`;
    if (task.status === 'failed' || task.status === 'cancelled') bar.hidden = true;
    const speed = active && task.status === 'downloading' ? updateSpeed(task) : 0;
    const eta = speed > 0 && total > downloaded ? (total - downloaded) / speed : 0;
    const meta = [
      task.source === 'backend' ? '推理后端' : task.source === 'huggingface' ? 'Hugging Face' : 'ModelScope',
      task.variant_label || task.variant,
      task.file ? task.file.split('/').pop() : null,
    ].filter(Boolean);
    const kind = task.status === 'done' ? 'success'
      : task.status === 'failed' ? 'error'
        : task.status === 'cancelled' ? 'muted'
          : 'accent';
    const actions = h('div', { class: 'download-actions' });
    if (active) {
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '取消',
        onClick: () => void cancel(task),
      }));
    } else {
      if (task.status !== 'done') {
        actions.append(h('button', {
          class: 'btn sm primary', type: 'button', text: '重试',
          onClick: () => void retry(task),
        }));
      }
      actions.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '移除',
        onClick: () => void remove(task),
      }));
    }
    return h('article', { class: 'download-card', dataset: { task: task.id } },
      h('div', { class: 'download-head' },
        h('div', { class: 'brand-mark', style: 'width:30px;height:30px;border-radius:8px', html: icon(task.status === 'done' ? 'check' : 'download', 15) }),
        h('div', { style: 'min-width:0;flex:1' },
          h('div', { class: 'model-name', text: task.repo }),
          h('div', { class: 'model-id', text: meta.join(' · ') || task.model_id }),
        ),
        h('span', { class: `chip ${kind}`, text: statusLabels[task.status] ?? task.status }),
      ),
      bar,
      h('div', { class: 'download-stats faint' },
        h('span', { text: total > 0 ? `${formatBytes(downloaded)} / ${formatBytes(total)} · ${percent}%` : `${formatBytes(downloaded)} · 大小未知` }),
        speed > 0 ? h('span', { text: `${formatBytes(speed)}/s` }) : null,
        eta > 0 ? h('span', { text: `剩余 ${formatEta(eta)}` }) : null,
      ),
      task.message && task.status !== 'downloading' ? h('div', { class: `download-message ${task.status === 'failed' ? 'error' : 'faint'}`, text: task.message }) : null,
      task.path && task.status === 'done' ? h('div', { class: 'download-path mono faint', text: task.path }) : null,
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

  async function cancel(task) {
    try {
      await cancelDownload(task.id);
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function retry(task) {
    try {
      await removeDownload(task.id);
      if (task.kind === 'backend') {
        if (!task.backend) throw new Error('后端下载描述已丢失，请重新从「后端」页面下载');
        await startBackendDownload(task.backend);
      } else {
        await startDownload({ repo: task.repo, source: task.source, variant: task.variant });
      }
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function remove(task) {
    try {
      await removeDownload(task.id);
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  const unsubscribe = subscribeDownloads(render);
  void initDownloads().then(render).catch(() => {});
  root.append(h('div', { class: 'card' },
    h('div', { class: 'section-title' }, h('h2', { text: '下载任务' }), h('span', { class: 'spacer' }),
      h('span', { class: 'faint', text: '同一时间下载 1 个模型；其余任务按顺序排队' })),
    summary,
    list,
  ));
  return () => {
    unsubscribe();
    samples.clear();
  };
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—';
  if (seconds < 60) return `${Math.ceil(seconds)} 秒`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} 分钟`;
  return `${(seconds / 3600).toFixed(1)} 小时`;
}
