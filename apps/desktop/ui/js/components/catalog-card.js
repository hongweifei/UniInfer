//! Catalog card rendering and download-state projection.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { isTerminalStatus } from '../downloads.js';

export function createCatalogCard(entry, { onDownload, onCancel }) {
  const button = h('button', {
    class: 'btn sm primary', type: 'button', dataset: { action: 'download' },
    onClick: onDownload,
  }, h('span', { html: icon('download', 13) }), h('span', { text: '下载' }));
  const cancelButton = h('button', {
    class: 'btn sm ghost', type: 'button', text: '取消', hidden: true, dataset: { action: 'cancel' },
    onClick: onCancel,
  });
  const bar = h('div', { class: 'progress', hidden: true }, h('span'));
  const text = h('div', { class: 'faint catalog-progress-text', hidden: true });
  const card = h('article', { class: 'search-card', dataset: { repo: entry.id } },
    h('div', { class: 'search-card-head' },
      h('div', { class: 'brand-mark', style: 'width:30px;height:30px;border-radius:8px', html: icon('cloud', 15) }),
      h('div', { style: 'min-width:0;flex:1' },
        h('div', { class: 'model-name', text: entry.id }),
        h('div', { class: 'model-id', text: entry.task || '模型仓库' }),
      ),
      h('span', { class: 'chip', text: formatDownloads(entry.downloads) }),
    ),
    entry.tags?.length
      ? h('div', { class: 'model-tags' }, ...entry.tags.slice(0, 4).map((tag) => h('span', { class: 'chip', text: tag })))
      : null,
    h('div', { class: 'catalog-actions' }, cancelButton, button),
    bar,
    text,
  );
  return { card, button, cancelButton, bar, text };
}

export function applyTask(task, button, cancelButton, bar, text) {
  if (!button) return;
  if (!task) {
    button.disabled = false;
    button.textContent = '下载';
    if (cancelButton) cancelButton.hidden = true;
    if (bar) bar.hidden = true;
    if (text) text.hidden = true;
    return;
  }
  const active = !isTerminalStatus(task.status);
  const total = Number(task.total) || 0;
  const downloaded = Number(task.downloaded) || 0;
  const percent = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : 0;
  if (cancelButton) {
    cancelButton.hidden = !active;
    cancelButton.disabled = false;
  }
  if (task.status === 'failed') {
    button.disabled = false;
    button.textContent = '重试';
    if (bar) bar.hidden = true;
    if (text) {
      text.hidden = false;
      text.textContent = task.message ?? '下载失败';
    }
    return;
  }
  if (task.status === 'cancelled') {
    button.disabled = false;
    button.textContent = '重新下载';
    if (bar) bar.hidden = true;
    if (text) {
      text.hidden = false;
      text.textContent = '已取消';
    }
    return;
  }
  if (task.status === 'done') {
    button.disabled = true;
    button.textContent = '已安装';
    if (bar) bar.hidden = true;
    if (text) {
      text.hidden = false;
      text.textContent = '下载完成，已加入模型库';
    }
    return;
  }
  button.disabled = true;
  const labels = {
    queued: '排队中', preparing: '准备中…', downloading: `下载 ${percent}%`,
    paused: '已暂停', verifying: '校验中…', inspecting: '生成配置…', committing: '安装中…',
  };
  button.textContent = labels[task.status] ?? '下载中…';
  if (bar) {
    bar.hidden = false;
    bar.classList.toggle('determinate', total > 0);
    const fill = bar.querySelector('span');
    if (fill) fill.style.width = total > 0 ? `${percent}%` : '36%';
  }
  if (text) {
    text.hidden = false;
    const detail = task.status === 'queued'
      ? '等待其他下载完成'
      : task.status === 'preparing'
        ? '读取仓库文件'
        : task.status === 'verifying'
          ? '校验文件完整性'
          : task.status === 'inspecting'
            ? '生成模型配置'
            : task.status === 'committing'
              ? '安装到模型库'
              : task.file ? task.file.split('/').pop() : '';
    text.textContent = total > 0 ? `${detail} · ${percent}%` : detail;
  }
}

export function formatDownloads(value) {
  const count = Number(value) || 0;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}
