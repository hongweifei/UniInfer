import { h } from './dom.js';
import { icon } from './icons.js';
import { ApiClient } from './api.js';
import { loadState, saveState, newConversation, recentConversations } from './state.js';
import { isAudioModel, isChatModel, isImageModel, modelTitle, errorMessage } from './format.js';
import { toast, themeIcon } from './components/ui.js';
import { mountChat } from './views/chat.js';
import { mountTranscribe } from './views/transcribe.js';
import { mountImages } from './views/images.js';
import { mountModels } from './views/models.js';
import { mountSettings } from './views/settings.js';

const views = {
  chat: { title: '对话', subtitle: '和本地模型聊天，支持图片输入', mount: mountChat },
  transcribe: { title: '语音转写', subtitle: '把 WAV 音频转成可编辑文本', mount: mountTranscribe },
  images: { title: '图像生成', subtitle: '用提示词生成图片', mount: mountImages },
  models: { title: '模型', subtitle: '管理本地模型与设备', mount: mountModels },
  settings: { title: '设置', subtitle: '外观、对话与运行时', mount: mountSettings },
};

const navItems = [
  ['chat', '对话', 'chat'],
  ['transcribe', '语音转写', 'wave'],
  ['images', '图像生成', 'image'],
  ['models', '模型', 'models'],
  ['settings', '设置', 'settings'],
];

const app = document.getElementById('app');
const state = loadState();
let api = null;
let models = [];
let devices = [];
let currentCleanup = null;

function applyTheme() {
  if (state.theme === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
  window.__TAURI__?.core?.invoke('set_window_theme', { theme: state.theme }).catch(() => {});
}

function persist() {
  saveState(state);
}

function modelById(id) {
  return models.find((model) => model.id === id) ?? null;
}

function currentModel() {
  return modelById(state.selectedModel);
}

function modelRank(model) {
  const id = model.id.toLowerCase();
  let rank = 0;
  if (model.loaded) rank -= 40;
  if (model.route_error) rank += 100;
  if (id.includes('tiny')) rank += 60;
  else if (id.includes('0.6b')) rank += id.includes('onnx') ? 0 : 3;
  else if (id.includes('256m')) rank += 4;
  else if (id.includes('2b')) rank += 10;
  else if (id.includes('7b')) rank += 20;
  return rank;
}

function sortModels(list) {
  return [...list].sort((a, b) => modelRank(a) - modelRank(b) || a.id.localeCompare(b.id));
}

function modelsForView(view) {
  let list = models;
  if (view === 'chat') list = list.filter(isChatModel);
  else if (view === 'transcribe') list = list.filter(isAudioModel);
  else if (view === 'images') list = list.filter(isImageModel);
  return list.filter((model) => !model.route_error);
}

function chooseModelForView(view) {
  const candidates = modelsForView(view);
  if (!candidates.length) return;
  if (candidates.some((model) => model.id === state.selectedModel)) return;
  state.selectedModel = candidates[0].id;
  persist();
}

function sidebar() {
  const nav = h('nav', { class: 'nav', attrs: { 'aria-label': '主导航' } });
  for (const [id, label, iconName] of navItems) {
    nav.append(h('button', {
      class: `nav-item ${state.view === id ? 'active' : ''}`,
      type: 'button',
      onClick: () => navigate(id),
      title: label,
    }, h('span', { html: icon(iconName, 17) }), h('span', { text: label })));
  }
  const recent = h('div', { class: 'recent-list' });
  const conversations = recentConversations(state, 8);
  for (const conversation of conversations) {
    recent.append(h('button', {
      class: `recent-item ${conversation.id === state.activeId ? 'active' : ''}`,
      type: 'button',
      title: conversation.title,
      onClick: () => {
        state.activeId = conversation.id;
        state.view = 'chat';
        persist();
        renderShell();
        renderView();
      },
    }, h('span', { html: icon('chat', 13) }), h('span', { text: conversation.title })));
  }
  if (!conversations.length) recent.append(h('div', { class: 'faint', style: 'padding:6px 9px;font-size:11.5px', text: '还没有对话' }));
  const loaded = models.filter((model) => model.loaded).length;
  const deviceClasses = [...new Set(devices.map((device) => device.class))];
  return h('aside', { class: 'sidebar' },
    h('div', { class: 'brand' },
      h('div', { class: 'brand-mark', html: icon('sparkles', 18) }),
      h('div', { class: 'brand-text' }, h('strong', { text: 'UniInfer' }), h('span', { text: '本地统一推理' })),
    ),
    h('button', {
      class: 'btn primary', type: 'button',
      onClick: () => { newConversation(state, currentModel()?.id); state.view = 'chat'; persist(); renderShell(); renderView(); },
    }, h('span', { html: icon('plus', 15) }), h('span', { text: '新对话' })),
    nav,
    h('div', { class: 'sidebar-section' },
      h('div', { class: 'sidebar-heading' }, h('span', { text: '最近' }), h('span', { text: String(state.conversations.length) })),
      recent,
    ),
    h('div', { class: 'sidebar-footer' },
      h('div', { class: 'runtime-card' },
        h('div', { class: 'runtime-row' }, h('span', { class: 'status-dot' }), h('strong', { text: '本地运行时' })),
        h('div', { class: 'runtime-row runtime-detail' }, h('span', { text: `${loaded} 已加载 · ${models.length} 已配置` })),
        h('div', { class: 'runtime-row runtime-detail' }, h('span', { html: icon('cpu', 12) }), h('span', { text: deviceClasses.join(' / ') || '未检测设备' })),
      ),
    ),
  );
}

function topbar() {
  const meta = views[state.view];
  const candidates = modelsForView(state.view);
  const select = h('select', {
    class: 'select model-select',
    'aria-label': '选择模型',
    disabled: candidates.length === 0,
    onChange: (event) => setModel(event.target.value),
  });
  for (const model of candidates) {
    select.append(h('option', { value: model.id, text: modelTitle(model) + (model.loaded ? ' · 已加载' : '') }));
  }
  if (state.selectedModel) select.value = state.selectedModel;
  const actions = h('div', { class: 'topbar-actions' });
  if (state.view === 'chat') {
    actions.append(select, h('button', {
      class: 'btn btn-icon ghost', type: 'button', title: '新对话',
      onClick: () => { newConversation(state, currentModel()?.id); persist(); renderShell(); renderView(); },
      html: icon('plus', 17),
    }));
  } else {
    actions.append(h('span', { class: 'chip', text: '工作台' }));
  }
  actions.append(h('button', {
    class: 'btn btn-icon ghost', type: 'button', title: `主题：${themeLabel(state.theme)}`,
    attrs: { 'data-theme-toggle': '' },
    onClick: () => cycleTheme(),
    html: icon(themeIcon(state.theme), 17),
  }));
  return h('header', { class: 'topbar' },
    h('div', { class: 'topbar-title' }, h('h1', { text: meta.title }), h('p', { text: meta.subtitle })),
    actions,
  );
}

function themeLabel(theme) {
  return theme === 'light' ? '浅色' : theme === 'dark' ? '深色' : '跟随系统';
}

function renderShell() {
  const main = h('main', { class: 'main', id: 'view' });
  app.replaceChildren(sidebar(), h('div', { class: 'workspace' }, topbar(), main));
}

function renderView() {
  currentCleanup?.();
  currentCleanup = null;
  const host = document.getElementById('view');
  if (!host) return;
  chooseModelForView(state.view);
  const meta = views[state.view];
  const ctx = createContext();
  currentCleanup = meta.mount(host, ctx) ?? null;
  document.dispatchEvent(new CustomEvent('uniinfer:model'));
}

function createContext() {
  return {
    state,
    api,
    models: () => models,
    devices: () => devices,
    currentModel,
    setModel,
    navigate,
    refreshModels,
    persist,
    setTheme,
    restart: () => window.__TAURI__?.core?.invoke('restart_app'),
    clearHistory: () => {
      state.conversations = [];
      state.activeId = null;
      persist();
      renderShell();
      renderView();
    },
    afterConversationChange: () => renderSidebarOnly(),
  };
}

function renderSidebarOnly() {
  const existing = app.querySelector('.sidebar');
  if (existing) existing.replaceWith(sidebar());
}

function setModel(id) {
  state.selectedModel = id;
  persist();
  const select = app.querySelector('.model-select');
  if (select) select.value = id;
  document.dispatchEvent(new CustomEvent('uniinfer:model'));
}

function setTheme(theme) {
  state.theme = theme;
  persist();
  applyTheme();
  const button = app.querySelector('[data-theme-toggle]');
  if (button) {
    button.innerHTML = icon(themeIcon(state.theme), 17);
    button.title = `主题：${themeLabel(state.theme)}`;
  }
}

function cycleTheme() {
  const next = state.theme === 'system' ? 'light' : state.theme === 'light' ? 'dark' : 'system';
  setTheme(next);
}

function navigate(view) {
  if (!views[view]) return;
  state.view = view;
  persist();
  chooseModelForView(view);
  renderShell();
  renderView();
}

async function refreshModels() {
  try {
    const [modelBody, deviceBody] = await Promise.all([api.models(), api.devices()]);
    models = sortModels(modelBody.data ?? []);
    devices = deviceBody.devices ?? [];
    chooseModelForView(state.view);
    renderShell();
    renderView();
    return models;
  } catch (error) {
    toast(`刷新模型失败：${errorMessage(error)}`, 'error');
    throw error;
  }
}

function fatal(desktop) {
  app.replaceChildren(h('div', { class: 'fatal' },
    h('div', { class: 'empty-icon', html: icon('alert', 26) }),
    h('h1', { text: '本地运行时没有启动' }),
    h('p', { class: 'muted', text: 'UniInfer 无法加载已配置的模型。检查模型清单或重启应用。' }),
    h('div', { class: 'code', text: `${desktop.error ?? '未知错误'}\n\n清单：${desktop.manifest ?? '未配置'}` }),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', type: 'button', text: '重试', onClick: () => location.reload() }),
      h('button', { class: 'btn', type: 'button', text: '重启应用', onClick: () => window.__TAURI__?.core?.invoke('restart_app') }),
    ),
  ));
}

async function boot() {
  applyTheme();
  let desktop;
  try {
    desktop = await window.__TAURI__.core.invoke('api_info');
  } catch (error) {
    fatal({ error: errorMessage(error), manifest: 'Tauri bridge unavailable' });
    return;
  }
  if (!desktop.info) {
    fatal(desktop);
    return;
  }
  api = new ApiClient(desktop.info);
  try {
    const [modelBody, deviceBody] = await Promise.all([api.models(), api.devices()]);
    models = sortModels(modelBody.data ?? []);
    devices = deviceBody.devices ?? [];
    chooseModelForView(state.view);
    renderShell();
    renderView();
  } catch (error) {
    fatal({ error: errorMessage(error), manifest: desktop.manifest });
  }
}

document.addEventListener('keydown', (event) => {
  const mod = event.ctrlKey || event.metaKey;
  if (mod && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    newConversation(state, currentModel()?.id);
    state.view = 'chat';
    persist();
    renderShell();
    renderView();
  }
  if (mod && event.key === ',') {
    event.preventDefault();
    navigate('settings');
  }
});
window.addEventListener('unhandledrejection', (event) => {
  toast(errorMessage(event.reason), 'error');
});

void boot();
