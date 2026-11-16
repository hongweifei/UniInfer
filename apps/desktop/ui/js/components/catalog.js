import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { toast } from './ui.js';

const invoke = (command, args) => window.__TAURI__?.core?.invoke(command, args);

export function mountCatalog(root, ctx) {
  const results = h('div', { class: 'search-results' });
  const pager = h('div', { class: 'pager' });
  const status = h('div', { class: 'faint', text: '搜索 ModelScope 或 Hugging Face 上的模型。' });
  const query = h('input', {
    class: 'input', type: 'search', placeholder: '搜索模型，例如 Qwen3、SmolVLM、Whisper',
    onKeyDown: (event) => { if (event.key === 'Enter') void search(); },
  });
  const source = h('select', { class: 'select', attrs: { 'aria-label': '模型源' } },
    h('option', { value: 'modelscope', text: 'ModelScope' }),
    h('option', { value: 'huggingface', text: 'Hugging Face' }),
  );
  source.addEventListener('change', () => { if (lastQuery) void search(1); });
  const searchButton = h('button', { class: 'btn primary', type: 'button', onClick: () => void search() },
    h('span', { html: icon('search', 14) }), h('span', { text: '搜索' }));
  const suggestions = h('div', { class: 'row' });
  for (const term of ['Qwen3', 'SmolVLM', 'Whisper']) {
    suggestions.append(h('button', {
      class: 'btn sm ghost', type: 'button', text: term,
      onClick: () => { query.value = term; void search(); },
    }));
  }
  const progress = new Map();
  let unlisten = null;
  let disposed = false;
  let currentPage = 1;
  let hasMore = false;
  let lastQuery = '';
  const PAGE_SIZE = 10;

  function sourceLabel() {
    return source.value === 'modelscope' ? 'ModelScope' : 'Hugging Face';
  }

  function renderPage(page, items) {
    clear(results);
    if (!items.length) {
      results.append(h('div', { class: 'notice', text: page === 1 ? '没有找到模型。换一个关键词或切换模型源。' : '这一页没有更多结果。' }));
    }
    for (const entry of items) {
      const button = h('button', {
        class: 'btn sm primary', type: 'button',
        onClick: () => {
          if (progress.get(entry.id)?.status === 'done') ctx.restart?.();
          else void install(entry, button);
        },
      }, h('span', { html: icon('download', 13) }), h('span', { text: '安装' }));
      const progressBar = h('div', { class: 'progress', hidden: true }, h('span'));
      const progressText = h('div', { class: 'faint catalog-progress-text', hidden: true });
      const card = h('article', { class: 'search-card', dataset: { repo: entry.id } },
        h('div', { class: 'search-card-head' },
          h('div', { class: 'brand-mark', style: 'width:30px;height:30px;border-radius:8px', html: icon('cloud', 15) }),
          h('div', { style: 'min-width:0;flex:1' },
            h('div', { class: 'model-name', text: entry.id }),
            h('div', { class: 'model-id', text: entry.task || '模型仓库' }),
          ),
          h('span', { class: 'chip', text: formatDownloads(entry.downloads) }),
        ),
        entry.tags?.length ? h('div', { class: 'model-tags' }, ...entry.tags.slice(0, 4).map((tag) => h('span', { class: 'chip', text: tag }))) : null,
        h('div', { class: 'catalog-actions' }, button),
        progressBar, progressText,
      );
      results.append(card);
      const state = progress.get(entry.id);
      if (state) updateProgress(entry.id, state, button, progressBar, progressText);
    }
    pager.replaceChildren(
      h('button', {
        class: 'btn sm ghost', type: 'button', text: '上一页', disabled: page <= 1,
        onClick: () => void search(page - 1),
      }),
      h('span', { class: 'faint', text: `第 ${page} 页` }),
      h('button', {
        class: 'btn sm ghost', type: 'button', text: '下一页', disabled: !hasMore,
        onClick: () => void search(page + 1),
      }),
    );
  }

  async function search(page = 1) {
    const text = query.value.trim();
    if (!text) return;
    lastQuery = text;
    if (page === 1) {
      currentPage = 1;
      hasMore = false;
    }
    status.textContent = `正在搜索 ${sourceLabel()}…`;
    clear(results);
    pager.replaceChildren();
    try {
      const result = await invoke('search_models', {
        query: text,
        source: source.value,
        page,
        pageSize: PAGE_SIZE,
      });
      if (disposed) return;
      currentPage = result.page ?? page;
      hasMore = Boolean(result.has_more);
      status.textContent = result.total
        ? `${result.total} 个结果 · ${sourceLabel()}`
        : `${result.items?.length ?? 0} 个结果 · ${sourceLabel()}`;
      renderPage(currentPage, result.items ?? []);
    } catch (error) {
      status.textContent = error?.message ?? String(error);
      toast(status.textContent, 'error');
    }
  }

  async function install(entry, button) {
    button.disabled = true;
    button.textContent = '准备下载…';
    progress.set(entry.id, { status: 'downloading', file: '', downloaded: 0, total: 1 });
    const card = results.querySelector(`[data-repo="${CSS.escape(entry.id)}"]`);
    const bar = card?.querySelector('.progress');
    const text = card?.querySelector('.catalog-progress-text');
    updateProgress(entry.id, progress.get(entry.id), button, bar, text);
    status.textContent = `正在下载 ${entry.id}，首次下载可能需要一些时间…`;
    try {
      const result = await invoke('install_model', { repo: entry.id, source: source.value });
      await ctx.api.request('/v1/models/refresh', { method: 'POST' }).catch(() => {});
      await ctx.refreshModels();
      progress.set(entry.id, { status: 'done' });
      status.textContent = `已安装 ${result.id}`;
      toast(`${entry.id} 下载完成并已加入模型库`, 'success');
    } catch (error) {
      const message = error?.message ?? String(error);
      progress.set(entry.id, { status: 'error', message });
      updateProgress(entry.id, progress.get(entry.id), button, bar, text);
      status.textContent = message;
      toast(message, 'error');
    }
  }

  if (window.__TAURI__?.event?.listen) {
    window.__TAURI__.event.listen('model-download', (event) => {
      const data = event.payload;
      const state = progress.get(data.repo) ?? { status: 'downloading', downloaded: 0, total: data.total };
      progress.set(data.repo, { ...state, ...data });
      const card = results.querySelector(`[data-repo="${CSS.escape(data.repo)}"]`);
      if (!card) return;
      updateProgress(
        data.repo,
        progress.get(data.repo),
        card.querySelector('.catalog-actions .btn'),
        card.querySelector('.progress'),
        card.querySelector('.catalog-progress-text'),
      );
    }).then((fn) => { unlisten = fn; });
  }

  root.append(h('div', { class: 'card' },
    h('div', { class: 'section-title' }, h('h2', { text: '在线模型' }), h('span', { class: 'spacer' }),
      h('span', { class: 'faint', text: '下载后自动生成 manifest 与默认运行配置' })),
    h('div', { class: 'search-bar' }, query, source, searchButton),
    h('div', { class: 'row', style: 'margin-top:8px' }, h('span', { class: 'faint', text: '试试：' }), suggestions),
    status, results, pager,
  ));
  void search();

  return () => {
    disposed = true;
    if (unlisten) unlisten();
  };
}

function formatDownloads(value) {
  const count = Number(value) || 0;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  const units = ['KiB', 'MiB', 'GiB'];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size >= 10 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}
