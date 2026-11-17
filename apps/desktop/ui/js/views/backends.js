import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, errorMessage } from '../format.js';
import { toast, chip, confirmDialog } from '../components/ui.js';

const invoke = (command, args) => window.__TAURI__?.core?.invoke(command, args);

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

  const catalogInput = h('input', {
    class: 'input',
    placeholder: 'https://example.com/backends/catalog.json',
    onInput: () => updateControls(),
  });
  const builtinHost = h('div', { class: 'backend-grid' });
  const sidecarHost = h('div', { class: 'backend-grid' });
  const availableHost = h('div', { class: 'backend-grid' });
  const runningName = h('strong', { text: '读取中…' });
  const runningVersion = h('span', { class: 'faint', text: '' });
  const builtinCount = h('span', { class: 'filter-count', text: '' });
  const sidecarCount = h('span', { class: 'filter-count', text: '' });
  const availableCount = h('span', { class: 'filter-count', text: '' });
  const directoryLabel = h('code', { class: 'mono', text: '读取中…' });
  const errorBanner = h('div', { class: 'notice error', hidden: true });
  const restartBanner = h('div', { class: 'backend-restart', hidden: true });
  const refreshButton = h('button', {
    class: 'btn btn-icon ghost sm', type: 'button', title: '重新读取后端列表',
    onClick: () => void refresh(),
  }, h('span', { html: icon('refresh', 14) }));
  const readCatalogButton = h('button', {
    class: 'btn primary', type: 'button',
    onClick: () => void readCatalog(),
  }, h('span', { html: icon('cloud', 14) }), h('span', { text: '读取目录' }));
  const importButton = h('button', {
    class: 'btn', type: 'button',
    onClick: () => void importLocal(),
  }, h('span', { html: icon('folder', 14) }), h('span', { text: '导入本地后端' }));
  const useBuiltinButton = h('button', {
    class: 'btn sm', type: 'button', hidden: true,
    onClick: () => void useBuiltin(),
  }, h('span', { html: icon('cpu', 14) }), h('span', { text: '切回内置引擎' }));

  function updateControls() {
    refreshButton.disabled = loading || busy;
    readCatalogButton.disabled = loading || busy || !catalogInput.value.trim();
    importButton.disabled = busy;
  }

  function showError(message) {
    if (!message) {
      errorBanner.hidden = true;
      errorBanner.textContent = '';
      return;
    }
    errorBanner.hidden = false;
    errorBanner.textContent = message;
  }

  function showRestart() {
    const restart = h('button', {
      class: 'btn sm primary', type: 'button',
      onClick: () => ctx.restart(),
    }, h('span', { html: icon('refresh', 14) }), h('span', { text: '重启应用' }));
    restartBanner.replaceChildren(
      h('span', { html: icon('alert', 15) }),
      h('span', { text: '后端选择已保存，重启后生效。' }),
      restart,
    );
    restartBanner.hidden = false;
  }

  async function loadConfig() {
    try {
      const response = await ctx.api.config();
      const server = response.config?.server ?? {};
      catalogUrl = server.backend_catalog_url ?? '';
      catalogInput.value = catalogUrl;
    } catch {
      // A config can be absent in embedded or test environments.
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
          message = `后端目录不可用：${errorMessage(error)}`;
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

  async function install(packageInfo) {
    if (busy) return;
    busy = true;
    render();
    try {
      const result = await invoke('install_backend', { package: packageInfo });
      toast(`已安装 ${packageInfo.name} ${packageInfo.version}`, 'success');
      await refresh({ reloadCatalog: false });
      if (result?.restart_required) showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    } finally {
      busy = false;
      render();
    }
  }

  async function importLocal() {
    try {
      const path = await invoke('pick_backend_folder');
      if (!path) return;
      busy = true;
      render();
      const result = await invoke('import_backend', { path });
      toast(`已导入 ${result.id} ${result.version}`, 'success');
      await refresh({ reloadCatalog: false });
      showRestart();
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
      toast(`已选择 ${backend.name}，重启后生效`, 'success');
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function useBuiltin() {
    try {
      await ctx.api.activateBackend('builtin');
      await refresh({ reloadCatalog: false });
      toast('已切回内置引擎组，重启后生效', 'success');
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function remove(backend) {
    const ok = await confirmDialog({
      title: '删除后端？',
      message: `${backend.name} ${backend.version} 会从磁盘移除；如果它正在使用，请先重启应用。`,
      confirmLabel: '删除', danger: true,
    });
    if (!ok) return;
    try {
      await ctx.api.removeBackend(backend.id);
      await refresh({ reloadCatalog: false });
      toast('后端已删除', 'success');
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  function isRunning(backend) {
    return state.running?.id === backend.id && state.running?.version === backend.version;
  }

  function isSelected(backend) {
    return state.active?.id === backend.id && state.active?.version === backend.version;
  }

  function sidecarStatus(backend) {
    if (isRunning(backend)) return chip('运行中', 'success');
    if (isSelected(backend)) return chip('待重启', 'warn');
    return chip('已安装', 'muted');
  }

  function engineList(backend) {
    const engines = backend.engines ?? [];
    if (!engines.length) {
      return h('div', { class: 'backend-engine-empty', text: '此包未声明引擎信息' });
    }
    const single = engines.length === 1 && (engines[0].name || engines[0].id) === backend.name;
    return h('div', { class: 'backend-engine-list' },
      ...engines.map((engine) => h('div', { class: `backend-engine ${single ? 'single' : ''}` },
        single ? null : h('div', { class: 'backend-engine-main' },
          h('strong', { text: engine.name || engine.id }),
          h('span', { text: `${engine.runtime || 'runtime?'} · ${engine.id}` }),
        ),
        h('div', { class: 'backend-engine-tags' },
          ...(engine.formats ?? []).map((value) => h('span', { class: 'backend-tag', text: FORMAT_LABELS[value] ?? value })),
          ...(engine.modalities ?? []).map((value) => h('span', { class: 'backend-tag muted', text: MODALITY_LABELS[value] ?? value })),
          ...(engine.devices ?? []).map((value) => h('span', { class: 'backend-tag muted', text: value.toUpperCase() })),
        ),
        (engine.implementations ?? []).length
          ? h('div', { class: 'backend-engine-impl', text: engine.implementations.join(' · ') })
          : null,
      )),
    );
  }

  function builtinCard(backend) {
    const running = state.running?.kind === 'builtin';
    return h('div', { class: `card backend-card builtin ${running ? 'running' : ''}` },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon builtin' }, h('span', { html: icon('cpu', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: backend.name }),
          h('span', { class: 'backend-card-id', text: `${backend.id} · ${backend.version}` }),
        ),
        running ? chip('运行中', 'success') : chip('可用', 'muted'),
      ),
      h('p', { class: 'backend-description', text: backend.description || '内置推理引擎' }),
      engineList(backend),
      h('div', { class: 'backend-meta' },
        ...(backend.engines?.length ? [] : [h('span', { class: 'backend-meta-item' }, h('span', { html: icon('cpu', 12) }), (backend.devices ?? []).join(' / ') || '设备未知')]),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('monitor', 12) }), backend.platform || '平台未知'),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('box', 12) }), '随主程序'),
      ),
    );
  }

  function sidecarCard(backend) {
    const selected = isSelected(backend);
    const running = isRunning(backend);
    const actionLabel = running ? '正在运行' : selected ? '等待重启' : '使用此后端';
    return h('div', { class: `card backend-card ${selected ? 'selected' : ''}` },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon' }, h('span', { html: icon('box', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: backend.name }),
          h('span', { class: 'backend-card-id', text: `${backend.id} · ${backend.version}` }),
        ),
        sidecarStatus(backend),
      ),
      h('p', { class: 'backend-description', text: backend.description || '未提供描述' }),
      engineList(backend),
      h('div', { class: 'backend-meta' },
        ...(backend.engines?.length ? [] : [h('span', { class: 'backend-meta-item' }, h('span', { html: icon('cpu', 12) }), (backend.devices ?? []).join(' / ') || '设备未知')]),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('monitor', 12) }), backend.platform || '平台未知'),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('box', 12) }), formatBytes(backend.size_bytes ?? 0)),
      ),
      h('div', { class: 'backend-card-actions' },
        h('button', {
          class: `btn sm ${selected ? 'ghost' : 'primary'}`, type: 'button',
          disabled: selected || busy, onClick: () => void activate(backend),
        }, selected ? h('span', { html: icon('check', 13) }) : null, h('span', { text: actionLabel })),
        h('span', { class: 'spacer' }),
        h('button', {
          class: 'btn btn-icon ghost sm danger', type: 'button', title: '删除后端',
          onClick: () => void remove(backend),
        }, h('span', { html: icon('trash', 14) })),
      ),
    );
  }

  function availableCard(packageInfo) {
    const already = (state.backends ?? []).some((item) => item.id === packageInfo.id && item.version === packageInfo.version);
    return h('div', { class: 'card backend-card available' },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon remote' }, h('span', { html: icon('cloud', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: packageInfo.name }),
          h('span', { class: 'backend-card-id', text: `${packageInfo.id} · ${packageInfo.version}` }),
        ),
        chip('可下载', 'accent'),
      ),
      h('p', { class: 'backend-description', text: packageInfo.description || '未提供描述' }),
      engineList(packageInfo),
      h('div', { class: 'backend-meta' },
        ...(packageInfo.engines?.length ? [] : [h('span', { class: 'backend-meta-item' }, h('span', { html: icon('cpu', 12) }), (packageInfo.devices ?? []).join(' / ') || '设备未知')]),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('monitor', 12) }), packageInfo.platform || '平台未知'),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('box', 12) }), `${packageInfo.files?.length ?? 0} 个文件`),
      ),
      h('div', { class: 'backend-card-actions' },
        h('button', {
          class: 'btn sm primary', type: 'button', disabled: busy || already,
          onClick: () => void install(packageInfo),
        }, h('span', { html: icon('download', 14) }), h('span', { text: already ? '已安装' : '下载安装' })),
      ),
    );
  }

  function emptyBackend(title, description, iconName = 'box') {
    return h('div', { class: 'backend-empty' },
      h('div', { class: 'backend-empty-icon', html: icon(iconName, 20) }),
      h('strong', { text: title }),
      h('p', { text: description }),
    );
  }

  function render() {
    updateControls();
    clear(builtinHost);
    clear(sidecarHost);
    clear(availableHost);
    const all = state.backends ?? [];
    const builtins = all.filter((backend) => backend.kind === 'builtin');
    const sidecars = all.filter((backend) => backend.kind !== 'builtin');
    const runningBuiltin = state.running?.kind === 'builtin';
    const builtinNames = builtins.map((backend) => backend.name).join(' · ');
    runningName.textContent = runningBuiltin ? '内置引擎组' : state.running?.id ?? '未运行';
    runningVersion.textContent = runningBuiltin
      ? `${builtins.length} 个内置引擎${builtinNames ? ` · ${builtinNames}` : ''}`
      : state.running?.version ?? '';
    runningVersion.title = runningVersion.textContent;
    builtinCount.textContent = `${builtins.length} 个`;
    sidecarCount.textContent = `${sidecars.length} 个`;
    directoryLabel.textContent = state.directory || '未配置';
    useBuiltinButton.hidden = state.active?.id === 'builtin';

    if (loading) {
      builtinHost.append(h('div', { class: 'backend-loading', text: '正在读取内置引擎…' }));
      sidecarHost.append(h('div', { class: 'backend-loading', text: '正在读取外置后端…' }));
    } else {
      for (const backend of builtins) builtinHost.append(builtinCard(backend));
      for (const backend of sidecars) sidecarHost.append(sidecarCard(backend));
      if (!builtins.length) {
        builtinHost.append(emptyBackend('没有内置引擎', '当前构建未启用任何内置推理引擎。', 'cpu'));
      }
      if (!sidecars.length) {
        sidecarHost.append(emptyBackend('没有外置后端', '下载或导入 Sidecar 后端后会显示在这里。', 'box'));
      }
    }

    const installedKeys = new Set(all.map((backend) => `${backend.id}@${backend.version}`));
    const available = (catalog?.backends ?? [])
      .filter((packageInfo) => !installedKeys.has(`${packageInfo.id}@${packageInfo.version}`));
    availableCount.textContent = `${available.length} 个`;
    if (loading) {
      availableHost.append(h('div', { class: 'backend-loading', text: '正在读取目录…' }));
    } else if (!catalog) {
      availableHost.append(emptyBackend(
        '尚未读取后端目录',
        '填写目录地址后点击“读取目录”，即可发现可下载的推理后端。',
        'cloud',
      ));
    } else if (!available.length) {
      availableHost.append(emptyBackend(
        '没有新的后端',
        '目录中的后端都已安装，或目录暂时为空。',
        'check',
      ));
    } else {
      for (const packageInfo of available) availableHost.append(availableCard(packageInfo));
    }
  }

  function renderPage() {
    return h('div', { class: 'page backend-view' },
      h('div', { class: 'backend-hero' },
        h('div', { class: 'backend-hero-text' },
          h('span', { class: 'backend-eyebrow', text: 'RUNTIME' }),
          h('h2', { text: '推理后端' }),
          h('p', { text: '内置引擎与外置 Sidecar 统一管理。选择一个后端，重启后切换推理运行时。' }),
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
            h('h2', { text: '后端目录' }),
            h('p', { text: '从远程 JSON 目录发现可下载的后端包。' }),
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
          importButton,
        ),
        h('div', { class: 'backend-path' },
          h('span', { class: 'backend-path-label', text: '安装位置' }),
          directoryLabel,
        ),
      ),
      h('section', { class: 'backend-section' },
        h('div', { class: 'backend-section-head' },
          h('div', {},
            h('h2', { text: '内置引擎' }),
            h('p', { text: '随主程序编译分发，始终作为可用能力。' }),
          ),
          h('span', { class: 'spacer' }),
          useBuiltinButton,
          builtinCount,
        ),
        builtinHost,
      ),
      h('section', { class: 'backend-section' },
        h('div', { class: 'backend-section-head' },
          h('div', {},
            h('h2', { text: '外置后端' }),
            h('p', { text: '独立安装的 Sidecar 后端，激活后优先处理其声明的引擎与模态。' }),
          ),
          h('span', { class: 'spacer' }),
          sidecarCount,
        ),
        sidecarHost,
      ),
      h('section', { class: 'backend-section' },
        h('div', { class: 'backend-section-head' },
          h('div', {},
            h('h2', { text: '可下载' }),
            h('p', { text: '来自后端目录的 Sidecar 后端包。' }),
          ),
          h('span', { class: 'spacer' }),
          availableCount,
        ),
        availableHost,
      ),
    );
  }

  root.append(renderPage());
  void (async () => {
    await loadConfig();
    await refresh();
  })();

  return () => {};
}
