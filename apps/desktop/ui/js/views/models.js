import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { formatBytes, isAudioModel, isImageModel, errorMessage, friendlyRouteError, capabilitySummary } from '../format.js';
import { modelCategories, modelTags, MODEL_CATEGORIES } from '../model-meta.js';
import { toast, emptyState, chip, setBusy, openModal, confirmDialog } from '../components/ui.js';
import { mountCatalog } from '../components/catalog.js';

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
    const tags = [...new Set(ctx.models().flatMap((model) => modelTags(model)))].sort((a, b) => a.localeCompare(b, 'zh-CN'));
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
        h('strong', { text: '设备' }), h('span', { text: (model.devices ?? []).join(' / ') || '待放置' }),
      ),
      routeError ? h('div', { class: 'notice warn', text: friendlyRouteError(routeError.message) }) : null,
      h('div', { class: 'model-card-actions' }, ...actions),
    );
  }

  function renderList() {
    const models = ctx.models().filter((model) => {
      if (filter !== 'all' && !modelCategories(model).includes(filter)) return false;
      if (tagFilter !== 'all' && !modelTags(model).includes(tagFilter)) return false;
      if (formatFilter !== 'all' && model.format !== formatFilter) return false;
      if (loadFilter === 'loaded' && !model.loaded) return false;
      if (loadFilter === 'idle' && model.loaded) return false;
      if (loadFilter === 'error' && !model.route_error) return false;
      if (!query) return true;
      const text = [model.id, model.format, model.quantization, model.runtime, model.implementation, ...(model.inputs ?? []), ...(model.outputs ?? []), ...modelTags(model)].join(' ').toLowerCase();
      return text.includes(query);
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
      h('div', { class: 'notice', text: '这里是模型的默认运行配置，保存后即时应用；下载模型会自动生成一份默认配置。' }),
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
  const subview = ctx.subview?.() ?? 'online';
  const headings = {
    online: ['在线模型', '搜索 ModelScope 或 Hugging Face，下载后自动生成默认运行配置。'],
    installed: ['已安装模型', '管理本地模型、加载状态与运行配置。'],
    devices: ['设备', '查看运行时探测到的 CPU、GPU、NPU 与放置结果。'],
    import: ['导入模型', '从已有目录动态探测并生成运行配置；适合离线模型。'],
  };
  const [heading, description] = headings[subview] ?? headings.online;
  const onlineHost = h('div', { attrs: { id: 'online-models' }, hidden: subview !== 'online' });
  ctx.onModelDownloaded = () => toast('模型下载完成并已加入模型库', 'success');
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

  return () => onlineCleanup?.();
}
