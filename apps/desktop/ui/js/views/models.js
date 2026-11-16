import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, formatName, isAudioModel, isChatModel, isImageModel, modality, errorMessage, friendlyRouteError, capabilitySummary } from '../format.js';
import { toast, emptyState, chip, setBusy, openModal, confirmDialog } from '../components/ui.js';
import { mountCatalog } from '../components/catalog.js';

function filtersFor(model) {
  const filters = ['全部'];
  if (isChatModel(model)) filters.push('对话');
  if (model.inputs?.includes('image')) filters.push('视觉');
  if (isAudioModel(model)) filters.push('转写');
  if (isImageModel(model)) filters.push('图像');
  return filters;
}

export function mountModels(root, ctx) {
  let filter = '全部';
  let query = '';
  let formatFilter = 'all';
  let loadFilter = 'all';
  let inspection = null;
  let importing = false;

  const list = h('div', { class: 'model-grid' });
  const count = h('span', { class: 'faint' });
  const search = h('input', {
    class: 'input', type: 'search', placeholder: '搜索模型、格式或能力',
    onInput: (event) => { query = event.target.value.trim().toLowerCase(); renderList(); },
  });
  const formatSelect = h('select', { class: 'select filter-select', attrs: { 'aria-label': '按格式筛选' } },
    h('option', { value: 'all', text: '全部格式' }),
    h('option', { value: 'onnx', text: 'ONNX' }),
    h('option', { value: 'gguf', text: 'GGUF' }),
    h('option', { value: 'safetensors', text: 'Safetensors' }),
    h('option', { value: 'ggml', text: 'GGML' }),
  );
  formatSelect.addEventListener('change', () => { formatFilter = formatSelect.value; renderList(); });
  const loadSelect = h('select', { class: 'select filter-select', attrs: { 'aria-label': '按加载状态筛选' } },
    h('option', { value: 'all', text: '全部状态' }),
    h('option', { value: 'loaded', text: '已加载' }),
    h('option', { value: 'idle', text: '未加载' }),
    h('option', { value: 'error', text: '不可用' }),
  );
  loadSelect.addEventListener('change', () => { loadFilter = loadSelect.value; renderList(); });
  const tabs = h('div', { class: 'row' });
  const deviceList = h('div', { class: 'device-list' });
  const importPath = h('input', { class: 'input', placeholder: '模型目录或 manifest.toml 路径' });
  const inspectButton = h('button', { class: 'btn', type: 'button', onClick: () => void inspect() },
    h('span', { html: icon('folder', 14) }), h('span', { text: '检查' }));
  const importButton = h('button', { class: 'btn primary', type: 'button', disabled: true, onClick: () => void importModel() },
    h('span', { html: icon('plus', 14) }), h('span', { text: '导入模型' }));
  const importResult = h('div', { class: 'notice', text: '选择一个包含 manifest.toml 的模型目录。路径需要位于配置根目录内。' });
  const promoteButton = h('button', {
    class: 'btn', type: 'button', hidden: true,
    onClick: () => window.__TAURI__?.core?.invoke('restart_app'),
  }, h('span', { html: icon('refresh', 14) }), h('span', { text: '重启以加载' }));

  const folderButton = h('button', { class: 'btn btn-icon', type: 'button', title: '选择目录', onClick: () => void pickFolder(), html: icon('folder', 15) });

  async function pickFolder() {
    try {
      const path = await window.__TAURI__?.core?.invoke('pick_model_folder');
      if (path) importPath.value = path;
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  function renderTabs() {
    clear(tabs);
    for (const name of ['全部', '对话', '视觉', '转写', '图像']) {
      tabs.append(h('button', {
        class: `btn sm ${filter === name ? '' : 'ghost'}`,
        type: 'button', text: name,
        onClick: () => { filter = name; renderTabs(); renderList(); },
      }));
    }
  }

  function modelCard(model) {
    const loaded = Boolean(model.loaded);
    const routeError = model.route_error;
    const imageOnly = (model.outputs ?? []).includes('image') && !(model.outputs ?? []).includes('text');
    const useButton = h('button', {
      class: 'btn sm primary', type: 'button', disabled: Boolean(routeError),
      title: routeError ? friendlyRouteError(routeError.message) : '使用此模型',
      onClick: () => {
        ctx.setModel(model.id);
        if (isImageModel(model)) ctx.navigate('images');
        else if (isAudioModel(model)) ctx.navigate('transcribe');
        else ctx.navigate('chat');
      },
    }, h('span', { html: icon('play', 13) }), h('span', { text: '使用' }));
    const loadButton = h('button', {
      class: 'btn sm', type: 'button', disabled: loaded || Boolean(routeError) || imageOnly,
      onClick: async () => {
        setBusy(loadButton, true, '加载中…');
        stateChip.textContent = '加载中';
        stateChip.className = 'chip';
        try {
          await ctx.api.loadModel(model.id);
          stateChip.textContent = '已加载';
          stateChip.className = 'chip success';
          toast(`${model.id} 已加载`, 'success');
          await ctx.refreshModels();
        } catch (error) {
          stateChip.textContent = '加载失败';
          stateChip.className = 'chip warn';
          toast(errorMessage(error), 'error');
        } finally {
          setBusy(loadButton, false);
        }
      },
    }, h('span', { html: icon('download', 13) }), h('span', { text: loaded ? '已加载' : '预加载' }));
    const unloadButton = h('button', {
      class: 'btn sm ghost', type: 'button', disabled: !loaded,
      onClick: async () => {
        setBusy(unloadButton, true, '卸载中…');
        try {
          await ctx.api.unloadModel(model.id);
          toast(`${model.id} 已卸载`, 'success');
          await ctx.refreshModels();
        } catch (error) {
          toast(errorMessage(error), 'error');
        } finally {
          setBusy(unloadButton, false);
        }
      },
    }, h('span', { text: '卸载' }));

    const stateChip = routeError
      ? chip('不可用', 'warn')
      : chip(loaded ? '已加载' : '未加载', loaded ? 'success' : '');
    return h('article', { class: 'model-card' },
      h('div', { class: 'model-card-head' },
        h('div', { class: 'brand-mark', style: 'width:32px;height:32px;border-radius:9px', html: icon(isAudioModel(model) ? 'wave' : isImageModel(model) ? 'image' : 'chat', 16) }),
        h('div', { style: 'min-width:0' },
          h('div', { class: 'model-name', text: model.id }),
          h('div', { class: 'model-id', text: model.engine ?? '未匹配引擎' }),
        ),
        h('span', { class: 'model-state' }, routeError ? chip('不可用', 'warn') : chip(loaded ? '已加载' : '按需', loaded ? 'success' : '')),
      ),
      h('div', { class: 'model-tags' },
        chip(formatName(model.format)),
        chip(model.quantization ?? 'unknown'),
        ...(model.inputs ?? []).map((item) => chip(modality(item), item === 'image' ? 'accent' : '')),
        ...(model.outputs ?? []).map((item) => chip(`→ ${modality(item)}`, item === 'image' ? 'accent' : '')),
      ),
      h('div', { class: 'model-kv' },
        h('strong', { text: '能力' }), h('span', { text: capabilitySummary(model) }),
        h('strong', { text: '上下文' }), h('span', { text: model.context_len ? `${model.context_len} tokens` : '—' }),
        h('strong', { text: '设备' }), h('span', { text: (model.devices ?? []).join(' / ') || '待放置' }),
      ),
      routeError ? h('div', { class: 'notice warn', text: friendlyRouteError(routeError.message) }) : null,
      h('div', { class: 'model-card-actions' }, useButton, loadButton, h('button', {
        class: 'btn sm ghost', type: 'button', onClick: () => void openConfig(model),
      }, h('span', { html: icon('settings', 13) }), h('span', { text: '配置' })), unloadButton, h('button', {
        class: 'btn sm danger', type: 'button', onClick: () => void removeModel(model),
      }, h('span', { html: icon('trash', 13) }), h('span', { text: '删除' }))),
    );
  }

  function renderList() {
    const models = ctx.models().filter((model) => {
      if (filter !== '全部' && !filtersFor(model).includes(filter)) return false;
      if (formatFilter !== 'all' && model.format !== formatFilter) return false;
      if (loadFilter === 'loaded' && !model.loaded) return false;
      if (loadFilter === 'idle' && model.loaded) return false;
      if (loadFilter === 'error' && !model.route_error) return false;
      if (!query) return true;
      const text = [model.id, model.format, model.quantization, model.runtime, model.implementation, ...(model.inputs ?? []), ...(model.outputs ?? [])].join(' ').toLowerCase();
      return text.includes(query);
    });
    count.textContent = `${models.length} / ${ctx.models().length}`;
    clear(list);
    if (!models.length) {
      list.append(emptyState({ iconName: 'models', title: '没有匹配的模型', description: '调整筛选，或在下方导入本地模型。' }));
      return;
    }
    for (const model of models) list.append(modelCard(model));
  }

  function renderDevices() {
    clear(deviceList);
    const devices = ctx.devices();
    if (!devices.length) {
      deviceList.append(h('div', { class: 'notice', text: '未检测到设备。' }));
      return;
    }
    for (const device of devices) {
      deviceList.append(h('div', { class: 'device-row' },
        h('span', { class: 'icon', html: icon('cpu', 15) }),
        h('div', { style: 'min-width:0;flex:1' },
          h('div', { text: `${device.class.toUpperCase()} · ${device.backend}` }),
          h('div', { class: 'device-id', text: device.id }),
        ),
        h('span', { class: 'faint', text: device.memory_bytes ? formatBytes(device.memory_bytes) : '内存未知' }),
      ));
    }
  }

  async function inspect() {
    const path = importPath.value.trim();
    if (!path) { toast('先填写模型目录', 'error'); return; }
    setBusy(inspectButton, true, '检查中…');
    importButton.disabled = true;
    try {
      inspection = await ctx.api.inspectModel(path);
      const manifest = inspection.manifest;
      importResult.className = 'notice';
      clear(importResult);
      importResult.append(
        h('div', { class: 'row' }, h('strong', { text: manifest.id }), chip(manifest.format), chip(manifest.quantization)),
        h('div', { class: 'faint', text: `${manifest.runtime} · ${manifest.implementation}` }),
        h('div', { style: 'margin-top:6px', text: `能力：${capabilitySummary(manifest)}` }),
        inspection.route?.supported
          ? h('div', { class: 'faint', text: `引擎：${inspection.route.engine} · ${(inspection.route.placement?.devices ?? []).map((item) => item.class).join(' / ') || '待放置'}` })
          : h('div', { class: 'notice warn', text: friendlyRouteError(inspection.route?.message ?? '当前运行时无法加载这个模型') }),
      );
      importButton.disabled = !inspection.route?.supported;
    } catch (error) {
      inspection = null;
      importResult.className = 'notice error';
      importResult.textContent = errorMessage(error);
    } finally {
      setBusy(inspectButton, false);
    }
  }

  async function importModel() {
    if (!inspection || importing) return;
    importing = true;
    setBusy(importButton, true, '导入中…');
    try {
      const result = await ctx.api.importModel(importPath.value.trim());
      importResult.className = 'notice';
      importResult.textContent = '模型已写入配置。重启应用后即可在列表中看到。';
      promoteButton.hidden = false;
      toast('模型已导入', 'success');
      if (result.restart_required) promoteButton.hidden = false;
    } catch (error) {
      importResult.className = 'notice error';
      importResult.textContent = errorMessage(error);
      toast(importResult.textContent, 'error');
    } finally {
      importing = false;
      setBusy(importButton, false);
    }
  }

  async function openConfig(model) {
    let manifest;
    try {
      const data = await ctx.api.request(`/v1/models/${encodeURIComponent(model.id)}/manifest`);
      manifest = data.manifest;
    } catch (error) {
      toast(errorMessage(error), 'error');
      return;
    }
    const placement = manifest.placement ?? {};
    const policy = h('select', { class: 'select' },
      ['auto', 'pinned', 'strict'].map((value) => h('option', { value, text: { auto: '自动', pinned: '固定', strict: '严格' }[value] })),
    );
    policy.value = placement.policy ?? 'auto';
    const prefers = new Set(placement.prefer ?? []);
    const prefRow = h('div', { class: 'row' });
    for (const [value, label] of [['cpu', 'CPU'], ['gpu', 'GPU'], ['npu', 'NPU']]) {
      const box = h('input', { type: 'checkbox', checked: prefers.has(value), onChange: (event) => {
        if (event.target.checked) prefers.add(value); else prefers.delete(value);
      } });
      prefRow.append(h('label', { class: 'row', style: 'gap:5px;font-size:12px;color:var(--muted)' }, box, label));
    }
    const fallback = h('input', { type: 'checkbox', checked: placement.allow_cpu_fallback !== false });
    const context = h('input', { class: 'input', type: 'number', min: '1', value: String(manifest.context_len ?? 2048) });
    const body = h('div', { style: 'display:grid;gap:12px' },
      h('div', { class: 'notice', text: '这里是模型的默认运行配置。保存后重启应用生效；下载模型会自动生成一份默认配置。' }),
      h('div', { class: 'field' }, h('label', { text: '设备放置策略' }), policy),
      h('div', { class: 'field' }, h('label', { text: '优先设备' }), prefRow),
      h('label', { class: 'row', style: 'gap:7px;font-size:13px' }, fallback, h('span', { text: '允许回退到 CPU' })),
      h('div', { class: 'field' }, h('label', { text: '上下文长度' }), context),
    );
    openModal({
      title: `运行配置 · ${model.id}`,
      body,
      actions: [
        { label: '取消', kind: 'ghost' },
        {
          label: '保存配置', kind: 'primary',
          onClick: async () => {
            manifest.placement = {
              ...placement,
              policy: policy.value,
              prefer: [...prefers],
              allow_cpu_fallback: fallback.checked,
            };
            manifest.context_len = Math.max(1, Number(context.value) || 2048);
            try {
              const result = await ctx.api.request(`/v1/models/${encodeURIComponent(model.id)}/manifest`, {
                method: 'PUT', body: manifest,
              });
              toast(result.restart_required ? '配置已保存，重启后生效' : '配置已保存并即时应用', 'success');
              await ctx.refreshModels();
            } catch (error) {
              toast(errorMessage(error), 'error');
              return false;
            }
          },
        },
      ],
    });
  }

  async function removeModel(model) {
    const ok = await confirmDialog({
      title: '删除模型？',
      message: `${model.id} 的文件和配置会从本地删除，删除后需要重启应用。`,
      confirmLabel: '删除并重启',
      danger: true,
    });
    if (!ok) return;
    try {
      await ctx.api.unloadModel(model.id).catch(() => {});
      const result = await window.__TAURI__?.core?.invoke('delete_model', { id: model.id });
      toast(result?.removed ? '模型已删除，正在重启' : '模型配置已移除，正在重启', 'success');
      setTimeout(() => ctx.restart?.(), 800);
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  renderTabs();
  renderList();
  renderDevices();
  const subview = ctx.subview?.() ?? 'online';
  const headings = {
    online: ['在线模型', '搜索 ModelScope 或 Hugging Face，下载后自动生成默认运行配置。'],
    installed: ['已安装模型', '管理本地模型、加载状态与运行配置。'],
    devices: ['设备', '查看运行时探测到的 CPU、GPU、NPU 与放置结果。'],
    import: ['导入模型', '从已有目录导入 manifest；适合离线模型。'],
  };
  const [heading, description] = headings[subview] ?? headings.online;
  const onlineHost = h('div', { attrs: { id: 'online-models' }, hidden: subview !== 'online' });
  ctx.onModelDownloaded = () => toast('模型下载完成，配置已写入；点击“重启并启用”后可用', 'success');
  const onlineCleanup = mountCatalog(onlineHost, ctx);

  root.append(h('div', { class: 'models-view' }, h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h2', { text: heading }),
      h('p', { text: description }),
    ),
    onlineHost,
    h('div', { class: 'card', attrs: { id: 'installed-models' }, hidden: subview !== 'installed' },
      h('div', { class: 'section-title' }, h('h2', { text: '已安装模型' }), h('span', { class: 'spacer' }), count,
        h('button', { class: 'btn sm ghost', type: 'button', onClick: () => void ctx.refreshModels(), html: `${icon('refresh', 14)} 刷新` }),
      ),
      h('div', { class: 'toolbar', style: 'margin-bottom:12px' }, search, formatSelect, loadSelect, tabs),
      list,
    ),
    h('div', { class: 'card', attrs: { id: 'import-model' }, hidden: subview !== 'import' },
      h('div', { class: 'section-title' }, h('h2', { text: '导入模型' }), h('span', { class: 'spacer' }),
        promoteButton),
      h('div', { class: 'import-box' },
        h('div', { class: 'input-row' }, importPath, folderButton, inspectButton, importButton),
        importResult,
      ),
    ),
    h('div', { class: 'card', attrs: { id: 'devices' }, hidden: subview !== 'devices' },
      h('div', { class: 'section-title' }, h('h2', { text: '设备' }), h('span', { class: 'spacer' }),
        h('span', { class: 'faint', text: '运行时探测结果' })),
      deviceList,
    ),
  )));

  return () => onlineCleanup?.();
}
