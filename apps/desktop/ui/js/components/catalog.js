import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { toast, openModal, chip } from './ui.js';
import { formatBytes, formatName, onlineCategories, onlineTags, MODEL_CATEGORIES } from '../format.js';

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
  const categoryBar = h('div', { class: 'filter-chips', style: 'margin-top:10px' });
  const tagBar = h('div', { class: 'filter-tags', style: 'margin-top:8px' });
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
  let totalResults = null;
  let currentItems = [];
  let categoryFilter = 'all';
  let tagFilter = 'all';
  const planCache = new Map();
  const PAGE_SIZE = 10;

  function sourceLabel() {
    return source.value === 'modelscope' ? 'ModelScope' : 'Hugging Face';
  }

  function filteredItems(items) {
    return items.filter((entry) => {
      if (categoryFilter !== 'all' && !onlineCategories(entry).includes(categoryFilter)) return false;
      if (tagFilter !== 'all' && !onlineTags(entry).includes(tagFilter)) return false;
      return true;
    });
  }

  function renderFilters(items) {
    const categoryCounts = new Map();
    const tags = new Set();
    for (const entry of items) {
      for (const category of onlineCategories(entry)) {
        categoryCounts.set(category, (categoryCounts.get(category) ?? 0) + 1);
      }
      for (const tag of onlineTags(entry)) tags.add(tag);
    }
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
    const sortedTags = [...tags].sort((a, b) => a.localeCompare(b, 'zh-CN'));
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
    const visible = filteredItems(items);
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
      const button = h('button', {
        class: 'btn sm primary', type: 'button',
        onClick: () => void chooseVariant(entry, button),
      }, h('span', { html: icon('download', 13) }), h('span', { text: '下载' }));
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
      if (totalPages) target = Math.min(target, totalPages);
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

  async function search(page = 1) {
    const text = query.value.trim();
    if (!text) return;
    lastQuery = text;
    if (page === 1) {
      currentPage = 1;
      hasMore = false;
      currentItems = [];
      categoryFilter = 'all';
      tagFilter = 'all';
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
      totalResults = result.total ?? null;
      status.textContent = result.total
        ? `${result.total} 个结果 · ${sourceLabel()}`
        : `${result.items?.length ?? 0} 个结果 · ${sourceLabel()}`;
      renderPage(currentPage, result.items ?? []);
    } catch (error) {
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
    openVariantDialog(entry, plan, button);
  }

  function openVariantDialog(entry, plan, button) {
    let selectedKey = plan.default_variant ?? plan.variants[0]?.key;
    const body = h('div', { class: 'variant-list' },
      h('p', { class: 'faint', text: '同一仓库包含多个格式或量化文件。选择要下载的变体；推荐项已预选。' }),
      ...plan.variants.map((variant) => {
        const input = h('input', {
          type: 'radio',
          name: `variant-${entry.id}`,
          value: variant.key,
          checked: variant.key === selectedKey,
          onChange: () => {
            selectedKey = variant.key;
            for (const node of body.querySelectorAll('.variant-option')) {
              node.classList.toggle('selected', node.contains(input));
            }
          },
        });
        return h('label', {
          class: `variant-option ${variant.key === selectedKey ? 'selected' : ''}`,
        },
        input,
        h('div', { class: 'variant-option-main' },
          h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap' },
            h('strong', { text: variant.label }),
            variant.recommended ? chip('推荐', 'success') : null,
            chip(formatName(variant.format)),
            variant.quantization && variant.quantization !== 'unknown' ? chip(variant.quantization) : null,
          ),
          h('div', { class: 'faint', text: `${variant.files} 个文件 · ${formatBytes(variant.total)}` }),
          variant.source ? h('div', { class: 'variant-source mono faint', text: variant.source }) : null,
        ));
      }),
    );
    let dialog;
    dialog = openModal({
      title: `选择下载变体 · ${entry.id}`,
      body,
      actions: [
        { label: '取消', kind: 'ghost' },
        {
          label: '开始下载', kind: 'primary',
          onClick: () => {
            dialog.close();
            button.disabled = false;
            button.textContent = '下载';
            void install(entry, button, selectedKey);
            return false;
          },
        },
      ],
    });
  }

  async function install(entry, button, variant) {
    button.disabled = true;
    button.textContent = '准备下载…';
    progress.set(entry.id, { status: 'downloading', variant, file: '', downloaded: 0, total: 1 });
    const card = results.querySelector(`[data-repo="${CSS.escape(entry.id)}"]`);
    const bar = card?.querySelector('.progress');
    const text = card?.querySelector('.catalog-progress-text');
    updateProgress(entry.id, progress.get(entry.id), button, bar, text);
    status.textContent = `正在下载 ${entry.id}，首次下载可能需要一些时间…`;
    try {
      const result = await invoke('install_model', { repo: entry.id, source: source.value, variant });
      await ctx.api.request('/v1/models/refresh', { method: 'POST' }).catch(() => {});
      await ctx.refreshModels();
      progress.set(entry.id, { status: 'done', variant });
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
    categoryBar, tagBar, status, results, pager,
  ));
  void search();

  return () => {
    disposed = true;
    if (unlisten) unlisten();
  };
}

function updateProgress(_repo, state, button, bar, text) {
  if (!button) return;
  if (state?.status === 'error') {
    button.disabled = false;
    button.textContent = '重试';
    if (bar) bar.hidden = true;
    if (text) {
      text.hidden = false;
      text.textContent = state.message ?? '下载失败';
    }
    return;
  }
  if (state?.status === 'done') {
    button.disabled = true;
    button.textContent = '已安装';
    if (bar) bar.hidden = true;
    if (text) {
      text.hidden = false;
      text.textContent = '下载完成，已加入模型库';
    }
    return;
  }
  const total = Number(state?.total) || 0;
  const downloaded = Number(state?.downloaded) || 0;
  const percent = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : 0;
  button.disabled = true;
  button.textContent = `下载 ${percent}%`;
  if (bar) {
    bar.hidden = false;
    const fill = bar.querySelector('span');
    if (fill) fill.style.width = `${Math.max(4, percent)}%`;
  }
  if (text) {
    text.hidden = false;
    text.textContent = state?.file ? `${state.file} · ${percent}%` : `准备中 · ${percent}%`;
  }
}

function formatDownloads(value) {
  const count = Number(value) || 0;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}

