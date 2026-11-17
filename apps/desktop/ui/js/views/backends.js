import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, errorMessage } from '../format.js';
import { toast, chip, setBusy, confirmDialog } from '../components/ui.js';

const invoke = (command, args) => window.__TAURI__?.core?.invoke(command, args);

export function mountBackends(root, ctx) {
  let backends = { installed: [], active: null, directory: '' };
  let catalog = null;
  let catalogUrl = '';
  let loading = true;
  let busy = false;

  const status = h('div', { class: 'notice faint', text: '读取后端列表…' });
  const directoryLabel = h('div', { class: 'notice faint', text: '安装目录：读取中…' });
  const catalogInput = h('input', {
    class: 'input', placeholder: 'https://…/backends/catalog.json',
  });
  const installedHost = h('div', { class: 'backend-grid' });
  const activeLabel = h('span', { class: 'faint', text: '当前：内置引擎' });
  const availableHost = h('div', { class: 'backend-grid' });
  const refreshButton = h('button', {
    class: 'btn sm', type: 'button',
    onClick: () => void refresh(),
  }, h('span', { html: icon('refresh', 14) }), h('span', { text: '刷新' }));
  const importButton = h('button', {
    class: 'btn sm', type: 'button',
    onClick: () => void importLocal(),
  }, h('span', { html: icon('folder', 14) }), h('span', { text: '导入本地后端' }));

  async function loadConfig() {
    try {
      const response = await ctx.api.config();
      const server = response.config?.server ?? {};
      catalogUrl = server.backend_catalog_url ?? '';
      catalogInput.value = catalogUrl;
    } catch {
      catalogInput.value = catalogUrl;
    }
  }

  async function refresh({ reloadCatalog = true } = {}) {
    loading = true;
    render();
    let message = '';
    try {
      backends = await ctx.api.backends();
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
      status.textContent = message || '后端列表已更新';
      render();
    }
  }

  async function saveCatalogUrl() {
    const next = catalogInput.value.trim();
    try {
      const current = await ctx.api.config();
      const body = {
        ...current.config,
        server: { ...(current.config?.server ?? {}), backend_catalog_url: next || null },
      };
      await ctx.api.saveConfig(body);
      catalogUrl = next;
      toast('后端目录已保存', 'success');
      await refresh();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function install(packageInfo) {
    if (busy) return;
    busy = true;
    render();
    try {
      const result = await invoke('install_backend', { package: packageInfo });
      toast(`已安装 ${packageInfo.name} ${packageInfo.version}，重启后生效`, 'success');
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
      toast(`已导入 ${result.id} ${result.version}，重启后生效`, 'success');
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
      toast(`已选择 ${backend.name} ${backend.version}，重启后生效`, 'success');
      await refresh({ reloadCatalog: false });
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  async function deactivate() {
    try {
      await ctx.api.deactivateBackend();
      toast('已切回内置引擎，重启后生效', 'success');
      await refresh({ reloadCatalog: false });
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
      toast('后端已删除，重启后生效', 'success');
      await refresh({ reloadCatalog: false });
      showRestart();
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  function showRestart() {
    const restart = h('button', {
      class: 'btn sm primary', type: 'button',
      onClick: () => ctx.restart(),
    }, h('span', { html: icon('refresh', 14) }), h('span', { text: '重启应用' }));
    status.replaceChildren(h('span', { text: '后端选择已保存；重启后使用新后端。' }), restart);
  }

  function installedCard(backend) {
    const active = backends.active?.id === backend.id && backends.active?.version === backend.version;
    return h('div', { class: 'card backend-card' },
      h('div', { class: 'row' },
        h('span', { html: icon('box', 18) }),
        h('div', { style: 'flex:1;min-width:0' },
          h('strong', { text: `${backend.name} ${backend.version}` }),
          h('div', { class: 'faint mono', text: backend.id }),
        ),
        active ? chip('使用中') : null,
      ),
      backend.description ? h('p', { class: 'muted', text: backend.description }) : null,
      h('div', { class: 'backend-meta' },
        h('span', { text: backend.platform || '未标注平台' }),
        h('span', { text: (backend.devices ?? []).join(' / ') || '设备未知' }),
        h('span', { text: formatBytes(backend.size_bytes ?? 0) }),
      ),
      h('div', { class: 'row' },
        h('button', {
          class: `btn sm ${active ? 'ghost' : 'primary'}`, type: 'button',
          disabled: active, onClick: () => void activate(backend),
        }, h('span', { text: active ? '当前后端' : '使用此后端' })),
        h('button', {
          class: 'btn sm danger ghost', type: 'button',
          onClick: () => void remove(backend),
        }, h('span', { html: icon('trash', 13) }), h('span', { text: '删除' })),
      ),
    );
  }

  function availableCard(packageInfo) {
    const already = backends.installed.some((item) => item.id === packageInfo.id && item.version === packageInfo.version);
    return h('div', { class: 'card backend-card' },
      h('div', { class: 'row' },
        h('span', { html: icon('cloud', 18) }),
        h('div', { style: 'flex:1;min-width:0' },
          h('strong', { text: `${packageInfo.name} ${packageInfo.version}` }),
          h('div', { class: 'faint mono', text: packageInfo.id }),
        ),
      ),
      packageInfo.description ? h('p', { class: 'muted', text: packageInfo.description }) : null,
      h('div', { class: 'backend-meta' },
        h('span', { text: packageInfo.platform || '未标注平台' }),
        h('span', { text: (packageInfo.devices ?? []).join(' / ') || '设备未知' }),
        h('span', { text: `${packageInfo.files?.length ?? 0} 个文件` }),
      ),
      h('button', {
        class: 'btn sm primary', type: 'button', disabled: busy || already,
        onClick: () => void install(packageInfo),
      }, h('span', { html: icon('download', 14) }), h('span', { text: already ? '已安装' : '下载安装' })),
    );
  }

  function render() {
    refreshButton.disabled = loading || busy;
    importButton.disabled = busy;
    clear(installedHost);
    clear(availableHost);
    activeLabel.textContent = backends.active
      ? `当前：${backends.active.id} ${backends.active.version}`
      : '当前：内置引擎';
    const installed = backends.installed ?? [];
    if (!loading && !installed.length) {
      installedHost.append(h('div', { class: 'notice faint', text: '还没有下载后端；当前使用内置引擎。' }));
    }
    for (const backend of installed) installedHost.append(installedCard(backend));
    const merged = new Map();
    for (const packageInfo of catalog?.backends ?? []) {
      merged.set(`${packageInfo.id}@${packageInfo.version}`, packageInfo);
    }
    for (const backend of installed) {
      const key = `${backend.id}@${backend.version}`;
      if (!merged.has(key)) {
        merged.set(key, {
          id: backend.id, name: backend.name, version: backend.version,
          description: '本地导入的后端包', platform: backend.platform,
          devices: backend.devices, files: [], installed: true,
        });
      }
    }
    const available = [...merged.values()].filter((packageInfo) => !packageInfo.installed
      && !installed.some((item) => item.id === packageInfo.id && item.version === packageInfo.version));
    if (!available.length && catalog) {
      availableHost.append(h('div', { class: 'notice faint', text: '目录中的后端都已安装。' }));
    }
    for (const packageInfo of available) availableHost.append(availableCard(packageInfo));
    directoryLabel.textContent = `安装目录：${backends.directory || '未配置'}`;
  }

  function renderPage() {
    return h('div', { class: 'page backend-view' },
      h('div', { class: 'page-head' },
        h('h2', { text: '推理后端' }),
        h('p', { text: '按需下载 Sidecar 后端；主程序保留 CPU 内置引擎，重启后切换。' }),
      ),
      h('div', { class: 'card' },
        h('div', { class: 'section-title' },
          h('h2', { text: '后端目录' }),
          h('span', { class: 'spacer' }),
          refreshButton,
        ),
        h('div', { class: 'row' },
          catalogInput,
          h('button', { class: 'btn', type: 'button', onClick: () => void saveCatalogUrl() }, h('span', { text: '保存' })),
          h('button', {
            class: 'btn primary', type: 'button', disabled: !catalogInput.value.trim(),
            onClick: async () => { catalogUrl = catalogInput.value.trim(); await refresh(); },
          }, h('span', { html: icon('cloud', 14) }), h('span', { text: '读取目录' })),
          importButton,
        ),
        directoryLabel,
      ),
      h('div', { class: 'section-title' }, h('h2', { text: '已安装' })),
      h('div', { class: 'row', style: 'margin-bottom:10px' },
        h('button', {
          class: 'btn sm', type: 'button', onClick: () => void deactivate(),
        }, h('span', { html: icon('cpu', 14) }), h('span', { text: '使用内置引擎' })),
        activeLabel,
      ),
      installedHost,
      h('div', { class: 'section-title', style: 'margin-top:22px' }, h('h2', { text: '可下载' })),
      availableHost,
      status,
    );
  }

  root.append(renderPage());
  void (async () => {
    await loadConfig();
    await refresh();
  })();

  return () => {};
}
