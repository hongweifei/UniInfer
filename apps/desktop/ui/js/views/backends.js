import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, errorMessage } from '../format.js';
import { toast, chip, confirmDialog } from '../components/ui.js';

const invoke = (command, args) => window.__TAURI__?.core?.invoke(command, args);

const CATEGORIES = [
  { id: 'language', title: '语言模型', subtitle: '聊天、补全、嵌入和重排序运行时。' },
  { id: 'audio', title: '语音', subtitle: '语音转写、语音合成和音频运行时。' },
  { id: 'image', title: '图像与视频', subtitle: '图像生成、图像编辑和视频生成运行时。' },
];

const FORMAT_LABELS = {
  onnx: 'ONNX',
  gguf: 'GGUF',
  safetensors: 'Safetensors',
  ggml: 'GGML',
};
const MODALITY_LABELS = {
  text: '文本',
  audio: '音频',
  image: '图像',
  video: '视频',
};

export function mountBackends(root, ctx) {
  let state = { backends: [], active: null, running: null, directory: '' };
  let catalog = null;
  let catalogUrl = '';
  let loading = true;
  let busy = false;

  const content = h('div', { class: 'backend-content' });
  const runningName = h('strong', { text: '读取中…' });
  const runningVersion = h('span', { class: 'faint', text: '' });
  const catalogInput = h('input', {
    class: 'input', placeholder: 'https://example.com/engines/catalog.json',
    onInput: () => updateControls(),
  });
  const directoryLabel = h('code', { class: 'mono', text: '读取中…' });
  const errorBanner = h('div', { class: 'notice error', hidden: true });
  const restartBanner = h('div', { class: 'backend-restart', hidden: true });
  const refreshButton = h('button', {
    class: 'btn btn-icon ghost sm', type: 'button', title: '重新读取推理引擎',
    onClick: () => void refresh(),
  }, h('span', { html: icon('refresh', 14) }));
  const readCatalogButton = h('button', {
    class: 'btn primary', type: 'button',
    onClick: () => void readCatalog(),
  }, h('span', { html: icon('cloud', 14) }), h('span', { text: '读取目录' }));

  function updateControls() {
    refreshButton.disabled = loading || busy;
    readCatalogButton.disabled = loading || busy || !catalogInput.value.trim();
  }

  function showError(message) {
    errorBanner.hidden = !message;
    errorBanner.textContent = message || '';
  }

  function showRestart() {
    const restart = h('button', {
      class: 'btn sm primary', type: 'button',
      onClick: () => ctx.restart(),
    }, h('span', { html: icon('refresh', 14) }), h('span', { text: '重启应用' }));
    restartBanner.replaceChildren(
      h('span', { html: icon('alert', 15) }),
      h('span', { text: '引擎选择已保存，重启后生效。' }),
      restart,
    );
    restartBanner.hidden = false;
  }

  async function loadConfig() {
    try {
      const response = await ctx.api.config();
      catalogUrl = response.config?.server?.backend_catalog_url ?? '';
      catalogInput.value = catalogUrl;
    } catch {
      catalogUrl = '';
    }
    updateControls();
  }

  async function refresh({ reloadCatalog = true } = {}) {
    loading = true;
    showError('');
    render();
    let message = '';
    try {
      state = await ctx.api.backends();
      if (reloadCatalog && catalogUrl) {
        try {
          catalog = await invoke('fetch_backend_catalog', { url: catalogUrl });
        } catch (error) {
          catalog = null;
          message = `引擎源不可用：${errorMessage(error)}`;
        }
      }
    } catch (error) {
      message = errorMessage(error);
    } finally {
      loading = false;
      showError(message);
      render();
    }
  }

  async function readCatalog() {
    const next = catalogInput.value.trim();
    if (!next) return;
    try {
      const current = await ctx.api.config();
      const body = {
        ...current.config,
        server: { ...(current.config?.server ?? {}), backend_catalog_url: next },
      };
      await ctx.api.saveConfig(body);
      catalogUrl = next;
      await refresh();
    } catch (error) {
      showError(errorMessage(error));
      toast(errorMessage(error), 'error');
    }
  }

  async function install(variant) {
    if (busy) return;
    busy = true;
    render();
    try {
      const result = await invoke('install_backend', { package: variant.backend });
      toast(`已安装 ${variant.engine.name || variant.vendor}`, 'success');
      await refresh({ reloadCatalog: false });
      if (result?.restart_required) showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    } finally {
      busy = false;
      render();
    }
  }

  async function activate(backend) {
    try {
      await ctx.api.activateBackend(backend.id, backend.version);
      await refresh({ reloadCatalog: false });
      toast(`已选择 ${backend.name}`, 'success');
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function remove(backend) {
    const ok = await confirmDialog({
      title: '卸载引擎？',
      message: `${backend.name} ${backend.version} 会从磁盘移除；如果它正在运行，请先重启应用。`,
      confirmLabel: '卸载', danger: true,
    });
    if (!ok) return;
    try {
      await ctx.api.removeBackend(backend.id);
      await refresh({ reloadCatalog: false });
      toast('引擎已卸载', 'success');
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  function categoryOf(engine) {
    if (CATEGORIES.some((category) => category.id === engine.category)) return engine.category;
    const modalities = engine.modalities ?? [];
    if (modalities.includes('image') || modalities.includes('video')) return 'image';
    if (modalities.includes('audio')) return 'audio';
    return 'language';
  }

  function variants() {
    const result = [];
    for (const backend of state.backends ?? []) {
      for (const engine of backend.engines ?? []) {
        result.push({
          backend,
          engine,
          vendor: engine.vendor || backend.name,
          source: backend.kind === 'builtin' ? 'builtin' : 'installed',
        });
      }
    }
    const installedKeys = new Set((state.backends ?? []).map((backend) => `${backend.id}@${backend.version}`));
    for (const backend of catalog?.backends ?? []) {
      if (installedKeys.has(`${backend.id}@${backend.version}`)) continue;
      for (const engine of backend.engines ?? []) {
        result.push({
          backend,
          engine,
          vendor: engine.vendor || backend.name,
          source: 'available',
        });
      }
    }
    return result;
  }

  function variantStatus(variant) {
    if (variant.source === 'builtin') {
      const running = state.running?.kind === 'builtin';
      return { label: running ? '运行中' : '可用', className: running ? 'success' : 'muted', dot: true };
    }
    if (variant.source === 'available') {
      return { label: '未安装', className: 'muted', dot: false };
    }
    const backend = variant.backend;
    if (state.running?.id === backend.id && state.running?.version === backend.version) {
      return { label: '运行中', className: 'success', dot: true };
    }
    if (state.active?.id === backend.id && state.active?.version === backend.version) {
      return { label: '待重启', className: 'warn', dot: true };
    }
    return { label: '已安装', className: 'success', dot: true };
  }

  function variantActions(variant) {
    const backend = variant.backend;
    if (variant.source === 'builtin') return [];
    if (variant.source === 'available') {
      return [h('button', {
        class: 'btn sm', type: 'button', disabled: busy,
        onClick: () => void install(variant),
      }, h('span', { html: icon('download', 13) }), h('span', { text: '安装' }))];
    }
    const selected = state.active?.id === backend.id && state.active?.version === backend.version;
    return [
      h('button', {
        class: 'btn sm ghost', type: 'button', disabled: selected || busy,
        onClick: () => void activate(backend),
      }, h('span', { text: selected ? '使用中' : '使用' })),
      h('button', {
        class: 'btn btn-icon ghost sm danger', type: 'button', title: '卸载',
        onClick: () => void remove(backend),
      }, h('span', { html: icon('trash', 13) })),
    ];
  }

  function variantTags(engine) {
    const tags = [
      ...(engine.formats ?? []).map((value) => FORMAT_LABELS[value] ?? value),
      ...(engine.modalities ?? []).map((value) => MODALITY_LABELS[value] ?? value),
      ...(engine.devices ?? []).map((value) => value.toUpperCase()),
    ];
    return h('div', { class: 'engine-variant-tags' },
      ...tags.map((tag, index) => h('span', {
        class: `engine-tag ${index < (engine.formats ?? []).length ? 'accent' : ''}`,
        text: tag,
      })),
    );
  }

  function variantRow(variant) {
    const { engine, backend } = variant;
    const status = variantStatus(variant);
    const version = backend.version === '0.0.0'
      ? '内置构建'
      : /^[vb]/.test(backend.version) ? backend.version : `v${backend.version}`;
    const versionNode = engine.homepage || backend.homepage
      ? h('a', {
          class: 'engine-version', href: engine.homepage || backend.homepage,
          target: '_blank', rel: 'noreferrer',
        }, h('span', { text: version }), h('span', { class: 'icon', html: icon('external', 10) }))
      : h('span', { class: 'engine-version', text: version });
    return h('div', { class: 'engine-variant' },
      h('div', { class: 'engine-variant-main' },
        h('div', { class: 'engine-variant-title' },
          h('strong', { text: engine.variant || (variant.source === 'builtin' ? '内置' : '默认') }),
          versionNode,
        ),
        variantTags(engine),
      ),
      h('div', { class: 'engine-variant-side' },
        h('span', { class: `engine-status ${status.className}` },
          status.dot ? h('i', { class: 'status-dot' }) : h('i', { class: 'status-dot hollow' }),
          h('span', { text: status.label }),
        ),
        h('div', { class: 'engine-variant-actions' }, ...variantActions(variant)),
      ),
    );
  }

  function vendorCard(vendor, entries) {
    const iconName = vendor.toLowerCase().includes('whisper')
      ? 'wave'
      : vendor.toLowerCase().includes('candle') ? 'sparkles' : 'box';
    return h('div', { class: 'vendor-card' },
      h('div', { class: 'vendor-brand' },
        h('span', { class: 'vendor-mark', html: icon(iconName, 20) }),
        h('strong', { text: vendor }),
      ),
      entries[0]?.engine.description
        ? h('p', { class: 'vendor-description', text: entries[0].engine.description })
        : null,
      h('div', { class: 'engine-variant-list' }, ...entries.map(variantRow)),
    );
  }

  function render() {
    updateControls();
    clear(content);
    const all = variants();
    const running = state.running;
    runningName.textContent = running
      ? (running.kind === 'builtin' ? '内置引擎组' : running.id)
      : '未启动';
    runningVersion.textContent = running
      ? (running.kind === 'builtin' ? `${countBuiltinEngines()} 个内置引擎` : `v${running.version}`)
      : '';
    directoryLabel.textContent = state.directory || '未配置';

    if (loading) {
      content.append(h('div', { class: 'backend-loading', text: '正在读取推理引擎…' }));
      return;
    }

    for (const category of CATEGORIES) {
      const entries = all.filter((variant) => categoryOf(variant.engine) === category.id);
      if (!entries.length) continue;
      const vendors = new Map();
      for (const entry of entries) {
        if (!vendors.has(entry.vendor)) vendors.set(entry.vendor, []);
        vendors.get(entry.vendor).push(entry);
      }
      content.append(h('section', { class: 'backend-category' },
        h('div', { class: 'backend-category-head' },
          h('div', {},
            h('h2', { text: category.title }),
            h('p', { text: category.subtitle }),
          ),
          h('span', { class: 'filter-count', text: `${vendors.size} 个项目 · ${entries.length} 个变体` }),
        ),
        h('div', { class: 'vendor-grid' },
          ...[...vendors.entries()].map(([vendor, variants]) => vendorCard(vendor, variants)),
        ),
      ));
    }

    if (!all.length) {
      content.append(h('div', { class: 'backend-empty' },
        h('div', { class: 'backend-empty-icon', html: icon('box', 20) }),
        h('strong', { text: '还没有可用的推理引擎' }),
        h('p', { text: '配置引擎源后，这里会列出当前设备可安装的官方推理引擎。' }),
      ));
    }
  }

  function countBuiltinEngines() {
    return (state.backends ?? [])
      .filter((backend) => backend.kind === 'builtin')
      .reduce((sum, backend) => sum + (backend.engines?.length ?? 0), 0);
  }

  function renderPage() {
    root.append(h('div', { class: 'page backend-view' },
      h('div', { class: 'backend-hero' },
        h('div', { class: 'backend-hero-text' },
          h('span', { class: 'backend-eyebrow', text: 'RUNTIME' }),
          h('h2', { text: '推理后端' }),
          h('p', { text: '安装和更新此设备上可用的推理引擎。' }),
        ),
        h('div', { class: 'backend-running' },
          h('div', { class: 'backend-running-label' },
            h('span', { class: 'status-dot' }),
            h('span', { text: '当前运行' }),
          ),
          runningName,
          runningVersion,
        ),
      ),
      restartBanner,
      errorBanner,
      h('section', { class: 'card backend-source' },
        h('div', { class: 'section-title' },
          h('div', {},
            h('h2', { text: '引擎源' }),
            h('p', { text: '从官方引擎目录读取可用版本和硬件变体。' }),
          ),
          h('span', { class: 'spacer' }),
          refreshButton,
        ),
        h('div', { class: 'backend-source-row' },
          h('div', { class: 'field backend-url-field' },
            h('label', { text: '目录 URL' }),
            catalogInput,
          ),
          readCatalogButton,
        ),
        h('div', { class: 'backend-path' },
          h('span', { class: 'backend-path-label', text: '安装位置' }),
          directoryLabel,
        ),
      ),
      content,
    ));
  }

  renderPage();
  void (async () => {
    await loadConfig();
    await refresh();
  })();

  return () => {};
}
