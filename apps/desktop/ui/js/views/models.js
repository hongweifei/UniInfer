import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, isAudioModel, isImageModel, errorMessage, friendlyRouteError, capabilitySummary } from '../format.js';
import { modelCategories, modelTags, MODEL_CATEGORIES, unsupportedUseReason } from '../model-meta.js';
import { filterModels, modelTagOptions } from '../model-filters.js';
import { toast, emptyState, chip, setBusy, openModal, confirmDialog } from '../components/ui.js';
import { mountCatalog } from '../components/catalog.js';
import { mountDownloads } from './downloads.js';
import { backendChoices, preferredVariant, variantOptions, variantSupport } from '../model-backend.js';

export function mountModels(root, ctx) {
  let filter = 'all';
  let tagFilter = 'all';
  let query = '';
  let formatFilter = 'all';
  let loadFilter = 'all';
  let inspection = null;
  let importing = false;

  const list = h('div', { class: 'model-grid' });
  const count = h('span', { class: 'faint' });
  const search = h('input', {
    class: 'input', type: 'search', placeholder: '搜索模型、格式、能力或标签',
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
  const tagSelect = h('select', { class: 'select filter-select', attrs: { 'aria-label': '按标签筛选' } });
  tagSelect.addEventListener('change', () => { tagFilter = tagSelect.value; renderList(); });
  const tabs = h('div', { class: 'filter-chips' });
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
    const models = ctx.models();
    clear(tabs);
    const all = h('button', {
      class: `btn sm ${filter === 'all' ? '' : 'ghost'}`,
      type: 'button',
      onClick: () => { filter = 'all'; renderTabs(); renderList(); },
    }, h('span', { text: '全部' }), h('span', { class: 'filter-count', text: String(models.length) }));
    tabs.append(all);
    for (const category of MODEL_CATEGORIES) {
      const count = models.filter((model) => modelCategories(model).includes(category.id)).length;
      if (category.id === 'other' && !count) continue;
      tabs.append(h('button', {
        class: `btn sm ${filter === category.id ? '' : 'ghost'}`,
        type: 'button',
        onClick: () => { filter = category.id; renderTabs(); renderList(); },
      }, h('span', { text: category.label }), h('span', { class: 'filter-count', text: String(count) })));
    }
  }

  function renderTagSelect() {
    const tags = modelTagOptions(ctx.models());
    if (tagFilter !== 'all' && !tags.includes(tagFilter)) tagFilter = 'all';
    clear(tagSelect);
    tagSelect.append(h('option', { value: 'all', text: '全部标签' }));
    for (const tag of tags) tagSelect.append(h('option', { value: tag, text: tag }));
    tagSelect.value = tagFilter;
  }

  function modelCard(model) {
    const loaded = Boolean(model.loaded);
    const routeError = model.route_error;
    const imageOnly = (model.outputs ?? []).includes('image') && !(model.outputs ?? []).includes('text');
    const unsupported = unsupportedUseReason(model);
    const useButton = h('button', {
      class: 'btn sm primary', type: 'button', disabled: Boolean(routeError) || Boolean(unsupported),
      title: routeError ? friendlyRouteError(routeError.message) : unsupported ?? '使用此模型',
      onClick: () => {
        ctx.setModel(model.id);
        if (isImageModel(model)) ctx.openMedia('images');
        else if (isAudioModel(model)) ctx.openMedia('transcribe');
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
    function closeMenu() {
      moreMenu?.removeAttribute('open');
    }
    const configItem = h('button', {
      class: 'model-menu-item', type: 'button',
      onClick: () => { closeMenu(); void openConfig(model); },
    }, h('span', { html: icon('settings', 14) }), h('span', { text: '运行配置' }));
    const unloadItem = loaded ? h('button', {
      class: 'model-menu-item', type: 'button',
      onClick: async () => {
        closeMenu();
        try {
          await ctx.api.unloadModel(model.id);
          toast(`${model.id} 已卸载`, 'success');
          await ctx.refreshModels();
        } catch (error) {
          toast(errorMessage(error), 'error');
        }
      },
    }, h('span', { html: icon('x', 14) }), h('span', { text: '卸载' })) : null;
    const deleteItem = h('button', {
      class: 'model-menu-item danger', type: 'button',
      onClick: () => { closeMenu(); void removeModel(model); },
    }, h('span', { html: icon('trash', 14) }), h('span', { text: '删除' }));
    let moreMenu;
    moreMenu = h('details', { class: 'model-more' },
      h('summary', {
        class: 'btn sm btn-icon ghost', title: '更多操作', attrs: { 'aria-label': '更多操作' },
      }, h('span', { html: icon('more', 16) })),
      h('div', { class: 'model-actions-menu' }, configItem, unloadItem, deleteItem),
    );
    const actions = [useButton];
    if (!loaded && !imageOnly && !routeError) actions.push(loadButton);
    actions.push(h('span', { class: 'spacer' }), moreMenu);

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
        ...modelTags(model).map((tag) => chip(tag, tag.startsWith('→') || tag === '图像' ? 'accent' : '')),
      ),
      h('div', { class: 'model-kv' },
        h('strong', { text: '能力' }), h('span', { text: capabilitySummary(model) }),
        h('strong', { text: '上下文' }), h('span', { text: model.context_len ? `${model.context_len} tokens` : '—' }),
        h('strong', { text: '后端' }), h('span', { text: model.backend?.id ? `${model.backend.id}${model.backend.variant && model.backend.variant !== 'auto' ? ` · ${model.backend.variant}` : ''}` : '自动' }),
        h('strong', { text: '设备' }), h('span', { text: (model.devices ?? []).join(' / ') || '待放置' }),
      ),
      routeError ? h('div', { class: 'notice warn', text: friendlyRouteError(routeError.message) }) : null,
      h('div', { class: 'model-card-actions' }, ...actions),
    );
  }

  function renderList() {
    const models = filterModels(ctx.models(), {
      category: filter, tag: tagFilter, format: formatFilter, load: loadFilter, query,
    });
    count.textContent = `${models.length} / ${ctx.models().length}`;
    clear(list);
    if (!models.length) {
      list.append(emptyState({ iconName: 'models', title: '没有匹配的模型', description: '调整筛选条件，或切换到「导入模型」添加本地模型。' }));
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
        h('span', { class: 'faint', text: device.memory_bytes
          ? `${device.unified_memory ? '共享主机 ' : ''}${formatBytes(device.memory_bytes)}${device.available_memory_bytes ? ` · ${formatBytes(device.available_memory_bytes)} 可用` : ''}`
          : '内存未知' }),
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
      await ctx.api.importModel(importPath.value.trim());
      importResult.className = 'notice';
      importResult.textContent = '模型已导入并加入模型库。';
      promoteButton.hidden = true;
      await ctx.api.request('/v1/models/refresh', { method: 'POST' }).catch(() => {});
      await ctx.refreshModels();
      toast('模型已导入', 'success');
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
    let sizeBytes = 0;
    try {
      const data = await ctx.api.request(`/v1/models/${encodeURIComponent(model.id)}/manifest`);
      manifest = data.manifest;
      sizeBytes = Number(data.size_bytes) || 0;
    } catch (error) {
      toast(errorMessage(error), 'error');
      return;
    }
    let installedBackends = [];
    try {
      installedBackends = (await ctx.api.backends()).backends ?? [];
    } catch (error) {
      toast(`读取后端列表失败：${errorMessage(error)}`, 'error');
    }
    const configuredBackend = manifest.backend ?? {};
    let backendId = configuredBackend.id || 'builtin';
    const choices = backendChoices(installedBackends);
    const builtinChoice = choices.find((choice) => choice.id === 'builtin');
    if (builtinChoice) {
      const engine = model.engine ? ` · ${model.engine}` : '';
      builtinChoice.label = `内置引擎${engine}`;
    }
    if (!choices.some((choice) => choice.id === backendId)) {
      choices.push({
        id: backendId,
        label: `${backendId}（未安装）`,
        variants: configuredBackend.variant && configuredBackend.variant !== 'auto'
          ? [configuredBackend.variant]
          : [],
      });
    }
    const backendSelect = h('select', { class: 'select', attrs: { 'aria-label': '推理后端' } });
    for (const choice of choices) {
      backendSelect.append(h('option', { value: choice.id, text: choice.label }));
    }
    backendSelect.value = backendId;
    const variantSelect = h('select', { class: 'select', attrs: { 'aria-label': '后端变体' } });
    const variantHint = h('div', { class: 'notice faint' });
    function rebuildVariants() {
      const choice = choices.find((item) => item.id === backendSelect.value);
      clear(variantSelect);
      variantSelect.replaceChildren();
      if (!choice || choice.id === 'builtin') {
        variantSelect.append(h('option', { value: 'auto', text: '内置' }));
        variantSelect.disabled = true;
        variantHint.textContent = '内置引擎由模型运行时自动选择。';
        return;
      }
      variantSelect.disabled = false;
      const variants = variantOptions(choice, configuredBackend.variant);
      const recommended = preferredVariant(ctx.devices(), variants, ctx.graphics());
      variantSelect.append(h('option', { value: 'auto', text: `自动（${recommended}）` }));
      for (const variant of variants) {
        const support = variantSupport(variant, ctx.devices(), ctx.graphics());
        const option = h('option', {
          value: variant,
          text: support.supported ? variant : `${variant}（当前设备不支持）`,
        });
        option.disabled = !support.supported;
        variantSelect.append(option);
      }
      const wanted = configuredBackend.variant || 'auto';
      variantSelect.value = [...variantSelect.options].some((option) => option.value === wanted)
        ? wanted
        : 'auto';
      const selectedSupport = variantSelect.value === 'auto'
        ? { supported: true, reason: '' }
        : variantSupport(variantSelect.value, ctx.devices(), ctx.graphics());
      variantHint.textContent = choice.variants.length
        ? selectedSupport.supported
          ? `自动会按当前设备优先选择 ${recommended}；可选变体来自已安装包。`
          : `当前选择的变体不可用：${selectedSupport.reason}。`
        : '该后端尚未安装，保存后模型在该后端安装前不可用。';
    }
    backendSelect.addEventListener('change', rebuildVariants);
    rebuildVariants();
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
    const offload = h('select', { class: 'select', attrs: { 'aria-label': '层卸载' } },
      h('option', { value: '', text: '默认（按引擎能力）' }),
      h('option', { value: 'auto', text: '自动（按空闲显存）' }),
      h('option', { value: '0', text: '全部 CPU' }),
      h('option', { value: '4', text: '固定 4 层到 GPU' }),
      h('option', { value: '8', text: '固定 8 层到 GPU' }),
      h('option', { value: '16', text: '固定 16 层到 GPU' }),
      h('option', { value: '999', text: '全部层到 GPU' }),
    );
    offload.value = placement.offload_layers ?? '';
    const kv = h('select', { class: 'select', attrs: { 'aria-label': 'KV cache 位置' } },
      h('option', { value: 'vram', text: 'VRAM' }),
      h('option', { value: 'ram', text: 'RAM' }),
      h('option', { value: 'auto', text: '自动' }),
    );
    kv.value = placement.kv_cache ?? 'vram';
    const supportsOffload = model.runtime === 'llama'
      || (model.runtime === 'onnx' && model.implementation === 'onnx.layered');
    if (!supportsOffload) {
      offload.disabled = true;
      kv.disabled = true;
      offload.value = '';
      kv.value = 'vram';
    }
    const devices = ctx.devices();
    const gpu = devices.find((device) => device.class === 'gpu' && !device.unified_memory)
      ?? devices.find((device) => device.class === 'gpu');
    const unified = Boolean(gpu?.unified_memory);
    const host = devices.find((device) => device.class === 'cpu');
    const memoryFree = Number(unified ? host?.available_memory_bytes : gpu?.available_memory_bytes) || 0;
    const tight = sizeBytes > 0 && memoryFree > 0 && sizeBytes > memoryFree;
    const recommendation = !supportsOffload
      ? '当前引擎没有层切分能力（需 llama.cpp 或 onnx.layered）。'
      : unified
        ? `统一内存 / 核显：GPU 与 CPU 共享同一物理内存池；模型 ${sizeBytes ? formatBytes(sizeBytes) : '体积未知'}，共享可用 ${memoryFree ? formatBytes(memoryFree) : '未知'}。建议自动层卸载 + KV 放 RAM，GPU/CPU 混合推理不额外占用显存。`
        : sizeBytes
          ? `${formatBytes(sizeBytes)} 模型；GPU 可用 ${memoryFree ? formatBytes(memoryFree) : '未知'}。建议：${tight ? '自动层卸载 + KV 放 RAM' : '自动层卸载 + KV 放 VRAM'}。`
          : '无法读取模型体积；建议使用自动层卸载。';
    const body = h('div', { style: 'display:grid;gap:12px' },
      h('div', { class: 'notice', text: '这里是模型的默认运行配置，保存后即时应用；不同大小、不同格式的模型应分别配置。' }),
      h('div', { class: 'notice faint', text: recommendation }),
      h('div', { class: 'field' }, h('label', { text: '推理后端' }), backendSelect),
      h('div', { class: 'field' }, h('label', { text: '后端变体' }), variantSelect),
      variantHint,
      h('div', { class: 'field' }, h('label', { text: '设备放置策略' }), policy),
      h('div', { class: 'field' }, h('label', { text: '优先设备' }), prefRow),
      h('label', { class: 'row', style: 'gap:7px;font-size:13px' }, fallback, h('span', { text: '允许回退到 CPU' })),
      h('div', { class: 'field' }, h('label', { text: '层卸载（llama.cpp / onnx.layered）' }), offload),
      h('div', { class: 'field' }, h('label', { text: 'KV cache 位置' }), kv),
      h('div', { class: 'field' }, h('label', { text: '上下文长度' }), context),
      h('div', { class: 'row' },
        h('button', { class: 'btn sm ghost', type: 'button', text: '使用推荐配置', onClick: () => {
          if (!supportsOffload) { toast('当前引擎不支持层卸载', 'error'); return; }
          offload.value = 'auto';
          kv.value = unified || tight ? 'ram' : 'vram';
          toast('已填入推荐配置，保存后生效', 'success', 1800);
        } }),
      ),
    );
    openModal({
      title: `运行配置 · ${model.id}`,
      body,
      actions: [
        { label: '取消', kind: 'ghost' },
        {
          label: '保存配置', kind: 'primary',
          onClick: async () => {
            const nextPlacement = {
              ...placement,
              policy: policy.value,
              prefer: [...prefers],
              allow_cpu_fallback: fallback.checked,
              kv_cache: kv.value,
            };
            if (offload.value) nextPlacement.offload_layers = offload.value;
            else delete nextPlacement.offload_layers;
            manifest.placement = nextPlacement;
            manifest.backend = backendSelect.value === 'builtin'
              ? { id: 'builtin' }
              : { id: backendSelect.value, variant: variantSelect.value || 'auto' };
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
      message: `${model.id} 的文件和配置会从本地删除，删除后立即从模型库移除。`,
      confirmLabel: '删除',
      danger: true,
    });
    if (!ok) return;
    try {
      await ctx.api.unloadModel(model.id).catch(() => {});
      const result = await window.__TAURI__?.core?.invoke('delete_model', { id: model.id });
      await ctx.api.request('/v1/models/refresh', { method: 'POST' }).catch(() => {});
      await ctx.refreshModels();
      toast(result?.removed ? '模型已删除' : '模型已从目录移除', 'success');
    } catch (error) {
      toast(errorMessage(error), 'error');
    }
  }

  renderTagSelect();
  renderTabs();
  renderList();
  renderDevices();
  const onModel = () => {
    renderTagSelect();
    renderTabs();
    renderList();
    renderDevices();
  };
  document.addEventListener('uniinfer:model', onModel);
  const subview = ctx.subview?.() ?? 'online';
  const headings = {
    online: ['在线模型', '搜索 ModelScope 或 Hugging Face，下载后自动生成默认运行配置。'],
    downloads: ['下载任务', '查看队列、进度、速度；支持取消、重试和清理。'],
    installed: ['已安装模型', '管理本地模型、加载状态与运行配置。'],
    devices: ['设备', '查看运行时探测到的 CPU、GPU、NPU 与放置结果。'],
    import: ['导入模型', '从已有目录动态探测并生成运行配置；适合离线模型。'],
  };
  const [heading, description] = headings[subview] ?? headings.online;
  const onlineHost = h('div', { attrs: { id: 'online-models' }, hidden: subview !== 'online' });
  const onlineCleanup = mountCatalog(onlineHost, ctx);
  const downloadHost = h('div', { attrs: { id: 'download-models' }, hidden: subview !== 'downloads' });
  const downloadCleanup = mountDownloads(downloadHost, ctx);

  root.append(h('div', { class: 'models-view' }, h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h2', { text: heading }),
      h('p', { text: description }),
    ),
    onlineHost,
    downloadHost,
    h('div', { class: 'card', attrs: { id: 'installed-models' }, hidden: subview !== 'installed' },
      h('div', { class: 'section-title' }, h('h2', { text: '已安装模型' }), h('span', { class: 'spacer' }), count,
        h('button', { class: 'btn sm ghost', type: 'button', onClick: () => void ctx.refreshModels(), html: `${icon('refresh', 14)} 刷新` }),
      ),
      h('div', { class: 'toolbar', style: 'margin-bottom:10px' }, search, formatSelect, loadSelect, tagSelect),
      h('div', { class: 'filter-chips', style: 'margin-bottom:12px' }, tabs),
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

  return () => {
    document.removeEventListener('uniinfer:model', onModel);
    onlineCleanup?.();
    downloadCleanup?.();
  };
}
