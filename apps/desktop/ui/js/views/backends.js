import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, errorMessage } from '../format.js';
import { toast, chip, confirmDialog } from '../components/ui.js';
import { startBackendDownload, subscribeDownloads, isTerminalStatus } from '../downloads.js';
import { variantSupport } from '../model-backend.js';

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
const ENGINE_SUMMARIES = {
  onnx: '运行 ONNX 模型，覆盖文本、语音、图像与视频处理。',
  candle: 'Rust 原生推理，支持 GGUF 与 Safetensors 模型。',
  llama: '运行 GGUF 文本模型，支持 CPU 与 GPU 混合推理。',
  'llama-mtmd': 'llama.cpp 多模态视觉语言引擎。',
  whisper: '语音转写引擎，支持 Whisper 模型。',
};
const TAG_LABELS = { ...FORMAT_LABELS, ...MODALITY_LABELS };

export function mountBackends(root, ctx) {
  let state = { backends: [], directory: '' };
  let catalog = null;
  let loading = true;
  let busy = false;

  const builtinHost = h('div', { class: 'backend-grid' });
  const sidecarHost = h('div', { class: 'backend-grid' });
  const availableHost = h('div', { class: 'backend-grid' });
  const builtinCount = h('span', { class: 'filter-count', text: '' });
  const sidecarCount = h('span', { class: 'filter-count', text: '' });
  const availableCount = h('span', { class: 'filter-count', text: '' });
  const errorBanner = h('div', { class: 'notice error', hidden: true });
  const refreshButton = h('button', {
    class: 'btn btn-icon ghost sm', type: 'button', title: '重新读取后端列表',
    onClick: () => void refresh(),
  }, h('span', { html: icon('refresh', 14) }));
  const downloadButton = h('button', {
    class: 'btn sm ghost', type: 'button',
    onClick: () => ctx.openDownloads(),
  }, h('span', { html: icon('download', 14) }), h('span', { text: '下载任务' }));


  function showError(message) {
    if (!message) {
      errorBanner.hidden = true;
      errorBanner.textContent = '';
      return;
    }
    errorBanner.hidden = false;
    errorBanner.textContent = message;
  }

  async function refresh({ reloadCatalog = true } = {}) {
    loading = true;
    showError('');
    render();
    let message = '';
    try {
      state = await ctx.api.backends();
      if (reloadCatalog) {
        try {
          catalog = await invoke('fetch_backend_catalog');
          if (catalog?.warnings?.length) {
            message = `部分引擎目录不可用：${catalog.warnings.join('；')}`;
          }
        } catch (error) {
          message = `官方引擎目录不可用：${errorMessage(error)}`;
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

  async function install(packageInfo) {
    if (busy) return;
    busy = true;
    render();
    try {
      await startBackendDownload(packageInfo);
      toast(`${packageInfo.name} ${packageInfo.engines?.[0]?.variant ?? ''} 已加入下载队列`, 'success');
    } catch (error) {
      toast(errorMessage(error), 'error');
    } finally {
      busy = false;
      render();
    }
  }

  async function remove(backend) {
    const ok = await confirmDialog({
      title: '删除后端？',
      message: `${backend.name} ${backend.version} 会从磁盘移除；正在加载它的模型需要卸载或重启后释放。`,
      confirmLabel: '删除', danger: true,
    });
    if (!ok) return;
    try {
      await ctx.api.removeBackend(backend.id);
      await refresh({ reloadCatalog: false });
      toast('后端已删除', 'success');
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  function capabilityRows(engine) {
    const groups = [
      ['格式', (engine.formats ?? []).map((value) => TAG_LABELS[value] ?? value)],
      ['内容', (engine.modalities ?? []).map((value) => MODALITY_LABELS[value] ?? value)],
      ['设备', (engine.devices ?? []).map((value) => value.toUpperCase())],
    ];
    return h('div', { class: 'backend-specs' },
      ...groups.filter(([, values]) => values.length).map(([label, values]) => h('div', { class: 'backend-spec' },
        h('span', { class: 'backend-spec-label', text: label }),
        h('span', { class: 'backend-spec-values', text: values.join(' · ') }),
      )),
    );
  }

  function engineList(backend) {
    const engines = backend.engines ?? [];
    if (!engines.length) return null;
    if (engines.length === 1) return capabilityRows(engines[0]);
    return h('div', { class: 'backend-engine-list' },
      ...engines.map((engine) => h('div', { class: 'backend-engine' },
        h('div', { class: 'backend-engine-main' },
          h('strong', { text: engine.variant || engine.name || engine.id }),
        ),
        capabilityRows(engine),
      )),
    );
  }

  function backendSummary(backend) {
    if (backend.kind === 'builtin') {
      const engine = backend.engines?.[0];
      const key = `${engine?.id ?? ''} ${backend.name}`.toLowerCase();
      if (ENGINE_SUMMARIES[engine?.id]) return ENGINE_SUMMARIES[engine.id];
      if (key.includes('onnx')) return ENGINE_SUMMARIES.onnx;
      if (key.includes('candle')) return ENGINE_SUMMARIES.candle;
      if (key.includes('whisper')) return ENGINE_SUMMARIES.whisper;
      if (key.includes('llama')) return ENGINE_SUMMARIES.llama;
      return '随应用内置的推理引擎。';
    }
    return backend.description || '官方推理引擎。';
  }

  function platformLabel(value) {
    const platform = (value || '').toLowerCase();
    if (platform.includes('windows')) return 'Windows';
    if (platform.includes('macos') || platform.includes('darwin')) return 'macOS';
    if (platform.includes('linux')) return 'Linux';
    return value || '当前平台';
  }

  function backendSubtitle(backend) {
    if (backend.kind === 'builtin') return '内置 · 随应用提供';
    const engine = backend.engines?.[0];
    const variant = engine?.variant ? ` · ${engine.variant}` : '';
    return backend.version === '0.0.0' ? `已安装${variant}` : `版本 ${backend.version}${variant}`;
  }

  function backendMeta(backend, kind) {
    if (kind === 'builtin') {
      return h('div', { class: 'backend-meta' },
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('monitor', 12) }), platformLabel(backend.platform)),
      );
    }
    if (kind === 'available') {
      return h('div', { class: 'backend-meta' },
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('monitor', 12) }), platformLabel(backend.platform)),
        h('span', { class: 'backend-meta-item' }, h('span', { html: icon('box', 12) }), `${backend.files?.length ?? 0} 个文件`),
      );
    }
    return h('div', { class: 'backend-meta' },
      h('span', { class: 'backend-meta-item' }, h('span', { html: icon('monitor', 12) }), platformLabel(backend.platform)),
      h('span', { class: 'backend-meta-item' }, h('span', { html: icon('box', 12) }), formatBytes(backend.size_bytes ?? 0)),
    );
  }

  function builtinCard(backend) {
    return h('div', { class: 'card backend-card builtin' },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon builtin' }, h('span', { html: icon('cpu', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: backend.name }),
          h('span', { class: 'backend-card-id', text: backendSubtitle(backend) }),
        ),
        chip('可用', 'muted'),
      ),
      h('p', { class: 'backend-description', text: backendSummary(backend) }),
      engineList(backend),
      backendMeta(backend, 'builtin'),
    );
  }

  function sidecarCard(packages) {
    const first = packages[0];
    const totalSize = packages.reduce((sum, item) => sum + (item.size_bytes ?? 0), 0);
    return h('div', { class: 'card backend-card' },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon' }, h('span', { html: icon('box', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: first.name }),
          h('span', { class: 'backend-card-id', text: `${packages.length} 个变体 · ${totalSize > 0 ? formatBytes(totalSize) : '大小未知'}` }),
        ),
        chip('已安装', 'muted'),
      ),
      h('p', { class: 'backend-description', text: backendSummary(first) }),
      engineList(first),
      h('div', { class: 'backend-engine-list' },
        ...packages.map((backend) => {
          const variant = backend.engines?.[0]?.variant || backend.version;
          const support = variantSupport(variant, ctx.devices(), ctx.graphics());
          return h('div', {
            class: `backend-engine with-action ${support.supported ? '' : 'unsupported'}`,
            title: support.reason,
          },
          h('div', { class: 'backend-engine-info' },
            h('div', { class: 'backend-engine-main' },
              h('strong', { text: variant }),
              h('span', { class: 'faint', text: `版本 ${backend.version}` }),
            ),
            !support.supported ? h('span', { class: 'backend-unsupported', text: support.reason }) : null,
          ),
            h('button', {
              class: 'btn btn-icon ghost sm danger', type: 'button', title: '删除这个变体',
              onClick: () => void remove(backend),
            }, h('span', { html: icon('trash', 14) })),
          );
        }),
      ),
      backendMeta(first, 'installed'),
    );
  }

  function availableCard(packages) {
    const first = packages[0];
    const installedKeys = new Set((state.backends ?? []).map((item) => `${item.id}@${item.version}`));
    const totalSize = packages.reduce((sum, item) => sum + (item.files ?? []).reduce((size, file) => size + (file.size ?? 0), 0), 0);
    return h('div', { class: 'card backend-card available' },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon remote' }, h('span', { html: icon('cloud', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: first.name }),
          h('span', { class: 'backend-card-id', text: `${packages.length} 个变体 · ${totalSize > 0 ? formatBytes(totalSize) : '大小未知'}` }),
        ),
        chip('可下载', 'accent'),
      ),
      h('p', { class: 'backend-description', text: backendSummary(first) }),
      engineList(first),
      h('div', { class: 'backend-engine-list' },
        ...packages.map((packageInfo) => {
          const installed = installedKeys.has(`${packageInfo.id}@${packageInfo.version}`);
          const variant = packageInfo.engines?.[0]?.variant || packageInfo.version;
          const support = variantSupport(variant, ctx.devices(), ctx.graphics());
          return h('div', {
            class: `backend-engine with-action ${support.supported ? '' : 'unsupported'}`,
            title: support.reason,
          },
            h('div', { class: 'backend-engine-info' },
              h('div', { class: 'backend-engine-main' },
                h('strong', { text: variant }),
                h('span', { class: 'faint', text: `版本 ${packageInfo.version} · ${packageInfo.files?.length ?? 0} 个文件` }),
              ),
              !support.supported ? h('span', { class: 'backend-unsupported', text: support.reason }) : null,
            ),
            h('button', {
              class: `btn sm ${installed || !support.supported ? 'ghost' : 'backend-download'}`, type: 'button',
              disabled: busy || installed || !support.supported,
              title: support.supported ? '' : support.reason,
              onClick: () => void install(packageInfo),
            }, h('span', { html: icon(installed ? 'check' : support.supported ? 'download' : 'x', 14) }),
              h('span', { text: installed ? '已安装' : support.supported ? '下载' : '不支持' })),
          );
        }),
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
    refreshButton.disabled = loading || busy;
    clear(builtinHost);
    clear(sidecarHost);
    clear(availableHost);
    const all = state.backends ?? [];
    const builtins = all.filter((backend) => backend.kind === 'builtin');
    const sidecars = all.filter((backend) => backend.kind !== 'builtin');
    const sidecarGroups = new Map();
    for (const backend of sidecars) {
      const group = sidecarGroups.get(backend.name) ?? [];
      group.push(backend);
      sidecarGroups.set(backend.name, group);
    }
    builtinCount.textContent = `${builtins.length} 个`;
    sidecarCount.textContent = `${sidecarGroups.size} 个`;

    if (loading) {
      builtinHost.append(h('div', { class: 'backend-loading', text: '正在读取内置引擎…' }));
      sidecarHost.append(h('div', { class: 'backend-loading', text: '正在读取外置后端…' }));
    } else {
      for (const backend of builtins) builtinHost.append(builtinCard(backend));
      for (const group of sidecarGroups.values()) sidecarHost.append(sidecarCard(group));
      if (!builtins.length) {
        builtinHost.append(emptyBackend('没有内置引擎', '当前构建未启用任何内置推理引擎。', 'cpu'));
      }
      if (!sidecars.length) {
        sidecarHost.append(emptyBackend('没有外置后端', '下载官方推理引擎后会显示在这里。', 'box'));
      }
    }

    const installedKeys = new Set(all.map((backend) => `${backend.id}@${backend.version}`));
    const available = (catalog?.backends ?? [])
      .filter((packageInfo) => !installedKeys.has(`${packageInfo.id}@${packageInfo.version}`));
    const availableGroups = new Map();
    for (const packageInfo of available) {
      const group = availableGroups.get(packageInfo.name) ?? [];
      group.push(packageInfo);
      availableGroups.set(packageInfo.name, group);
    }
    availableCount.textContent = `${availableGroups.size} 个引擎`;
    if (loading) {
      availableHost.append(h('div', { class: 'backend-loading', text: '正在读取目录…' }));
    } else if (!catalog) {
      availableHost.append(emptyBackend(
        '暂时没有下载项',
        '刷新后会从官方 release 读取当前设备可安装的推理引擎。',
        'cloud',
      ));
    } else if (!availableGroups.size) {
      availableHost.append(emptyBackend(
        '没有新的后端',
        '目录中的后端都已安装，或目录暂时为空。',
        'check',
      ));
    } else {
      for (const group of availableGroups.values()) availableHost.append(availableCard(group));
    }
  }

  function renderPage() {
    return h('div', { class: 'page backend-view' },
      h('div', { class: 'backend-hero' },
        h('div', { class: 'backend-hero-text' },
          h('span', { class: 'backend-eyebrow', text: 'RUNTIME' }),
          h('h2', { text: '推理后端' }),
          h('p', { text: '后端安装后即可被模型选用；模型各自配置后端与设备变体。' }),
        ),
        h('div', { class: 'backend-hero-side' },
          downloadButton,
          refreshButton,
        ),
      ),
      errorBanner,
      h('section', { class: 'backend-section', attrs: { id: 'backend-builtin' } },
        h('div', { class: 'backend-section-head' },
          h('div', {},
            h('h2', { text: '内置引擎' }),
            h('p', { text: '随主程序编译分发，始终作为可用能力。' }),
          ),
          h('span', { class: 'spacer' }),
          builtinCount,
        ),
        builtinHost,
      ),
      h('section', { class: 'backend-section', attrs: { id: 'backend-installed' } },
        h('div', { class: 'backend-section-head' },
          h('div', {},
            h('h2', { text: '外置后端' }),
            h('p', { text: '独立下载的官方推理引擎；是否使用由模型的运行配置决定。' }),
          ),
          h('span', { class: 'spacer' }),
          sidecarCount,
        ),
        sidecarHost,
      ),
      h('section', { class: 'backend-section', attrs: { id: 'backend-available' } },
        h('div', { class: 'backend-section-head' },
          h('div', {},
            h('h2', { text: '可下载' }),
            h('p', { text: '从官方 release 获取的可安装版本。' }),
          ),
          h('span', { class: 'spacer' }),
          availableCount,
        ),
        availableHost,
      ),
    );
  }

  root.append(renderPage());
  const refreshedBackends = new Set();
  const unsubscribeDownloads = subscribeDownloads((tasks) => {
    for (const task of tasks) {
      if (task.kind !== 'backend' || !isTerminalStatus(task.status) || refreshedBackends.has(task.id)) continue;
      refreshedBackends.add(task.id);
      if (task.status === 'done') void refresh({ reloadCatalog: false });
    }
  });
  void refresh();

  return () => {
    unsubscribeDownloads();
  };
}
