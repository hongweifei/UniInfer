import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { toast } from './ui.js';
import { openVariantDialog } from './variant-dialog.js';
import { applyTask, createCatalogCard } from './catalog-card.js';
import { MODEL_CATEGORIES } from '../model-meta.js';
import { clampPage, filterOnlineItems, onlineFacets } from '../catalog-utils.js';
import { cancelDownload, findDownload, isTerminalStatus, removeDownload, startDownload, subscribeDownloads } from '../downloads.js';

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
  source.addEventListener('change', () => {
    resetResults();
    if (lastQuery) void search(1);
  });
  const searchButton = h('button', { class: 'btn primary', type: 'button', onClick: () => void search() },
    h('span', { html: icon('search', 14) }), h('span', { text: '搜索' }));
  const categoryBar = h('div', { class: 'filter-chips', style: 'margin-top:10px' });
  const tagBar = h('div', { class: 'filter-tags', style: 'margin-top:8px' });
  const suggestions = h('div', { class: 'row' });
  for (const term of ['Qwen3', 'SmolVLM', 'Whisper']) {
    suggestions.append(h('button', {
      class: 'btn sm ghost', type: 'button', text: term,
      onClick: () => { query.value = term; void search(); },
    }));
  }
  let disposed = false;
  let searchSeq = 0;
  let currentPage = 1;
  let hasMore = false;
  let lastQuery = '';
  let totalResults = null;
  let currentItems = [];
  let categoryFilter = 'all';
  let tagFilter = 'all';
  const planCache = new Map();
  const PAGE_SIZE = 10;

  function sourceLabel() {
    return source.value === 'modelscope' ? 'ModelScope' : 'Hugging Face';
  }

  function renderFilters(items) {
    const { categories: categoryCounts, tags } = onlineFacets(items);
    if (categoryFilter !== 'all' && !categoryCounts.has(categoryFilter)) categoryFilter = 'all';
    if (tagFilter !== 'all' && !tags.has(tagFilter)) tagFilter = 'all';

    const filtersChanged = () => {
      renderPage(currentPage, currentItems);
    };
    clear(categoryBar);
    categoryBar.append(h('button', {
      class: `btn sm ${categoryFilter === 'all' ? '' : 'ghost'}`,
      type: 'button',
      onClick: () => { categoryFilter = 'all'; filtersChanged(); },
    }, h('span', { text: '全部分类' }), h('span', { class: 'filter-count', text: String(items.length) })));
    for (const category of MODEL_CATEGORIES) {
      const count = categoryCounts.get(category.id) ?? 0;
      if (!count) continue;
      categoryBar.append(h('button', {
        class: `btn sm ${categoryFilter === category.id ? '' : 'ghost'}`,
        type: 'button',
        onClick: () => { categoryFilter = category.id; filtersChanged(); },
      }, h('span', { text: category.label }), h('span', { class: 'filter-count', text: String(count) })));
    }
    if (categoryFilter !== 'all' || tagFilter !== 'all') {
      categoryBar.append(h('button', {
        class: 'btn sm ghost', type: 'button', text: '清除筛选',
        onClick: () => { categoryFilter = 'all'; tagFilter = 'all'; filtersChanged(); },
      }));
    }

    clear(tagBar);
    tagBar.append(h('span', { class: 'faint', text: '标签' }));
    const sortedTags = tags;
    tagBar.append(h('button', {
      class: `btn sm ${tagFilter === 'all' ? '' : 'ghost'}`,
      type: 'button', text: '全部',
      onClick: () => { tagFilter = 'all'; filtersChanged(); },
    }));
    for (const tag of sortedTags) {
      tagBar.append(h('button', {
        class: `btn sm ${tagFilter === tag ? '' : 'ghost'}`,
        type: 'button', text: tag,
        onClick: () => { tagFilter = tagFilter === tag ? 'all' : tag; filtersChanged(); },
      }));
    }
  }

  function renderPage(page, items) {
    currentItems = items;
    renderFilters(items);
    const visible = filterOnlineItems(items, { category: categoryFilter, tag: tagFilter });
    clear(results);
    if (!items.length) {
      results.append(h('div', { class: 'notice', text: page === 1 ? '没有找到模型。换一个关键词或切换模型源。' : '这一页没有更多结果。' }));
    } else if (!visible.length) {
      results.append(h('div', { class: 'notice' },
        h('span', { text: '当前筛选没有匹配结果。' }),
        h('button', {
          class: 'btn sm ghost', type: 'button', text: '清除筛选',
          onClick: () => { categoryFilter = 'all'; tagFilter = 'all'; renderPage(currentPage, currentItems); },
        }),
      ));
    }
    for (const entry of visible) {
      const { card, button, cancelButton, bar, text } = createCatalogCard(entry, {
        onDownload: () => void handleDownload(entry, button),
        onCancel: () => void handleCancel(entry),
      });
      results.append(card);
      applyTask(findDownload(entry.id, source.value), button, cancelButton, bar, text);
    }
    const totalPages = totalResults ? Math.max(1, Math.ceil(totalResults / PAGE_SIZE)) : null;
    const pageInput = h('input', {
      class: 'input page-jump-input',
      type: 'number',
      min: '1',
      max: totalPages ? String(totalPages) : null,
      value: String(page),
      attrs: { 'aria-label': '跳转页码' },
      onKeyDown: (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          goToPage();
        }
      },
    });
    const goToPage = () => {
      let target = Number.parseInt(pageInput.value, 10);
      if (!Number.isFinite(target) || target < 1) {
        pageInput.value = String(page);
        return;
      }
      target = clampPage(target, totalResults, PAGE_SIZE) ?? page;
      if (target !== page) void search(target);
    };
    pager.replaceChildren(
      h('button', {
        class: 'btn sm ghost', type: 'button', text: '上一页', disabled: page <= 1,
        onClick: () => void search(page - 1),
      }),
      h('span', { class: 'faint', text: `第 ${page} 页` }),
      h('button', {
        class: 'btn sm ghost', type: 'button', text: '下一页', disabled: !hasMore && (!totalPages || page >= totalPages),
        onClick: () => void search(page + 1),
      }),
      h('span', { class: 'faint', text: totalPages ? `/ 共 ${totalPages} 页` : '/ 页数未知' }),
      pageInput,
      h('button', {
        class: 'btn sm', type: 'button', text: '跳转',
        onClick: goToPage,
      }),
    );
  }

  function resetResults() {
    searchSeq += 1;
    currentItems = [];
    clear(results);
    clear(categoryBar);
    clear(tagBar);
    pager.replaceChildren();
  }

  async function search(page = 1) {
    const text = query.value.trim();
    const seq = ++searchSeq;
    if (!text) return;
    const requestedSource = source.value;
    const requestedLabel = sourceLabel();
    lastQuery = text;
    if (page === 1) {
      currentPage = 1;
      hasMore = false;
      currentItems = [];
      categoryFilter = 'all';
      tagFilter = 'all';
    }
    status.textContent = `正在搜索 ${requestedLabel}…`;
    clear(results);
    clear(categoryBar);
    clear(tagBar);
    pager.replaceChildren();
    try {
      const result = await invoke('search_models', {
        query: text,
        source: requestedSource,
        page,
        pageSize: PAGE_SIZE,
      });
      if (disposed || seq !== searchSeq) return;
      currentPage = result.page ?? page;
      hasMore = Boolean(result.has_more);
      totalResults = result.total ?? null;
      status.textContent = result.total
        ? `${result.total} 个结果 · ${requestedLabel}`
        : `${result.items?.length ?? 0} 个结果 · ${requestedLabel}`;
      renderPage(currentPage, result.items ?? []);
    } catch (error) {
      if (disposed || seq !== searchSeq) return;
      status.textContent = error?.message ?? String(error);
      toast(status.textContent, 'error');
    }
  }

  async function chooseVariant(entry, button) {
    const cacheKey = `${source.value}:${entry.id}`;
    let plan = planCache.get(cacheKey);
    if (!plan) {
      const previous = button.textContent;
      button.disabled = true;
      button.textContent = '读取变体…';
      try {
        plan = await invoke('plan_model', { repo: entry.id, source: source.value });
        planCache.set(cacheKey, plan);
      } catch (error) {
        const message = error?.message ?? String(error);
        button.disabled = false;
        button.textContent = previous;
        toast(message, 'error');
        return;
      }
    }
    button.disabled = false;
    button.textContent = '下载';
    const variants = plan.variants ?? [];
    if (!variants.length) {
      toast('这个仓库没有可下载的模型文件', 'error');
      return;
    }
    if (variants.length === 1) {
      void install(entry, button, variants[0].key);
      return;
    }
    openVariantDialog({ entry, plan, onInstall: (key) => { button.disabled = false; button.textContent = '下载'; void install(entry, button, key); } });
  }

  async function handleDownload(entry, button) {
    const task = findDownload(entry.id, source.value);
    if (task && !isTerminalStatus(task.status)) return;
    if (task && task.status !== 'done') {
      try {
        await removeDownload(task.id);
      } catch (error) {
        toast(error?.message ?? String(error), 'error');
        return;
      }
      await install(entry, button, task.variant);
      return;
    }
    await chooseVariant(entry, button);
  }

  async function handleCancel(entry) {
    const task = findDownload(entry.id, source.value);
    if (!task || isTerminalStatus(task.status)) return;
    try {
      await cancelDownload(task.id);
    } catch (error) {
      toast(error?.message ?? String(error), 'error');
    }
  }

  async function install(entry, button, variant) {
    button.disabled = true;
    button.textContent = '加入队列…';
    status.textContent = `已加入下载队列：${entry.id}`;
    try {
      const task = await startDownload({ repo: entry.id, source: source.value, variant });
      const card = results.querySelector(`[data-repo="${CSS.escape(entry.id)}"]`);
      if (card) {
        applyTask(
          task,
          card.querySelector('[data-action="download"]'),
          card.querySelector('[data-action="cancel"]'),
          card.querySelector('.progress'),
          card.querySelector('.catalog-progress-text'),
        );
      }
      toast(`${entry.id} 已加入下载队列`, 'success');
    } catch (error) {
      const message = error?.message ?? String(error);
      status.textContent = message;
      toast(message, 'error');
    }
  }

  const unsubscribe = subscribeDownloads(() => {
    for (const card of results.querySelectorAll('.search-card')) {
      const repo = card.dataset.repo;
      applyTask(
        findDownload(repo, source.value),
        card.querySelector('[data-action="download"]'),
        card.querySelector('[data-action="cancel"]'),
        card.querySelector('.progress'),
        card.querySelector('.catalog-progress-text'),
      );
    }
  });

  root.append(h('div', { class: 'card' },
    h('div', { class: 'section-title' }, h('h2', { text: '在线模型' }), h('span', { class: 'spacer' }),
      h('span', { class: 'faint', text: '下载后自动生成 manifest 与默认运行配置' })),
    h('div', { class: 'search-bar' }, query, source, searchButton),
    h('div', { class: 'row', style: 'margin-top:8px' }, h('span', { class: 'faint', text: '试试：' }), suggestions),
    categoryBar, tagBar, status, results, pager,
  ));
  void search();

  return () => {
    disposed = true;
    unsubscribe();
  };
}
