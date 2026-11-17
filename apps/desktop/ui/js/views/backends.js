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
const ENGINE_SUMMARIES = {
  onnx: '运行 ONNX 模型，覆盖文本、语音、图像与视频处理。',
  candle: 'Rust 原生推理，支持 GGUF 与 Safetensors 模型。',
  llama: '运行 GGUF 文本模型，支持 CPU 与 GPU 混合推理。',
  'llama-mtmd': 'llama.cpp 多模态视觉语言引擎。',
  whisper: '语音转写引擎，支持 Whisper 模型。',
};
const TAG_LABELS = { ...FORMAT_LABELS, ...MODALITY_LABELS };

export function mountBackends(root, ctx) {
  let state = { backends: [], active: null, running: null, directory: '' };
  let catalog = null;
  let loading = true;
  let busy = false;

  const builtinHost = h('div', { class: 'backend-grid' });
  const sidecarHost = h('div', { class: 'backend-grid' });
  const availableHost = h('div', { class: 'backend-grid' });
  const runningName = h('strong', { text: '读取中…' });
  const runningVersion = h('span', { class: 'faint', text: '' });
  const builtinCount = h('span', { class: 'filter-count', text: '' });
  const sidecarCount = h('span', { class: 'filter-count', text: '' });
  const availableCount = h('span', { class: 'filter-count', text: '' });
  const errorBanner = h('div', { class: 'notice error', hidden: true });
  const restartBanner = h('div', { class: 'backend-restart', hidden: true });
  const refreshButton = h('button', {
    class: 'btn btn-icon ghost sm', type: 'button', title: '重新读取后端列表',
    onClick: () => void refresh(),
  }, h('span', { html: icon('refresh', 14) }));
  const useBuiltinButton = h('button', {
    class: 'btn sm', type: 'button', hidden: true,
    onClick: () => void useBuiltin(),
  }, h('span', { html: icon('cpu', 14) }), h('span', { text: '切回内置引擎' }));


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
    const running = state.running?.kind === 'builtin';
    return h('div', { class: `card backend-card builtin ${running ? 'running' : ''}` },
      h('div', { class: 'backend-card-head' },
        h('div', { class: 'backend-icon builtin' }, h('span', { html: icon('cpu', 20) })),
        h('div', { class: 'backend-card-title' },
          h('strong', { text: backend.name }),
          h('span', { class: 'backend-card-id', text: backendSubtitle(backend) }),
        ),
        running ? chip('运行中', 'success') : chip('可用', 'muted'),
      ),
      h('p', { class: 'backend-description', text: backendSummary(backend) }),
      engineList(backend),
      backendMeta(backend, 'builtin'),
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
          h('span', { class: 'backend-card-id', text: backendSubtitle(backend) }),
        ),
        sidecarStatus(backend),
      ),
      h('p', { class: 'backend-description', text: backendSummary(backend) }),
      engineList(backend),
      backendMeta(backend, 'installed'),
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
          h('span', { class: 'backend-card-id', text: `版本 ${packageInfo.version}` }),
        ),
        chip('可下载', 'accent'),
      ),
      h('p', { class: 'backend-description', text: backendSummary(packageInfo) }),
      engineList(packageInfo),
      backendMeta(packageInfo, 'available'),
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
    refreshButton.disabled = loading || busy;
    clear(builtinHost);
    clear(sidecarHost);
    clear(availableHost);
    const all = state.backends ?? [];
    const builtins = all.filter((backend) => backend.kind === 'builtin');
    const sidecars = all.filter((backend) => backend.kind !== 'builtin');
    const runningBuiltin = state.running?.kind === 'builtin';
    runningName.textContent = runningBuiltin ? '内置引擎组' : state.running?.id ?? '未运行';
    runningVersion.textContent = runningBuiltin
      ? `${builtins.length} 个引擎`
      : state.running?.version ? `版本 ${state.running.version}` : '';
    builtinCount.textContent = `${builtins.length} 个`;
    sidecarCount.textContent = `${sidecars.length} 个`;
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
        sidecarHost.append(emptyBackend('没有外置后端', '下载官方推理引擎后会显示在这里。', 'box'));
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
        '暂时没有下载项',
        '刷新后会从官方 release 读取当前设备可安装的推理引擎。',
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
        h('div', { class: 'backend-hero-side' },
          h('div', { class: 'backend-running' },
            h('div', { class: 'backend-running-label' },
              h('span', { class: 'status-dot' }),
              h('span', { text: '当前运行' }),
            ),
            runningName,
            runningVersion,
          ),
          refreshButton,
        ),
      ),
      restartBanner,
      errorBanner,
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
            h('p', { text: '独立下载的官方推理引擎，激活后用于对应模型。' }),
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
  void refresh();

  return () => {};
}
