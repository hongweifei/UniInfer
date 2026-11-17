import { h } from './dom.js';
import { icon } from './icons.js';
import { ApiClient } from './api.js';
import { loadState, saveState, newConversation, deleteConversation, recentConversations } from './state.js';
import { isAudioModel, isChatModel, isImageModel, modelTitle, errorMessage } from './format.js';
import { toast, themeIcon, confirmDialog } from './components/ui.js';
import { deleteMedia } from './media-store.js';
import { initDownloads, subscribeDownloads, activeDownloads } from './downloads.js';
import { mountChat } from './views/chat.js';
import { mountModels } from './views/models.js';
import { mountMedia, mediaTools } from './views/media.js';
import { mountBackends } from './views/backends.js';
import { mountSettings } from './views/settings.js';

const views = {
  chat: { title: '对话', subtitle: '和本地模型聊天，支持图片输入', mount: mountChat },
  media: { title: '媒体', subtitle: '转写、生成与多模态工具', mount: mountMedia },
  models: { title: '模型', subtitle: '管理本地模型与设备', mount: mountModels },
  backends: { title: '后端', subtitle: '下载和管理推理后端', mount: mountBackends },
  settings: { title: '设置', subtitle: '外观、对话与运行时', mount: mountSettings },
};

const navItems = [
  ['chat', '对话', 'chat'],
  ['media', '媒体', 'sparkles'],
  ['models', '模型', 'models'],
  ['backends', '后端', 'box'],
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

function mediaTab() {
  return ['transcribe', 'images', 'tts', 'video'].includes(state.mediaTab) ? state.mediaTab : 'transcribe';
}

function modelsForView(view) {
  let list = models;
  if (view === 'chat') {
    list = list.filter(isChatModel);
  } else if (view === 'media') {
    const tab = mediaTab();
    if (tab === 'transcribe') list = list.filter(isAudioModel);
    else if (tab === 'images') list = list.filter(isImageModel);
    else return [];
  }
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
  const context = [];
  if (state.view === 'chat') {
    context.push(h('button', {
      class: 'btn primary', type: 'button',
      onClick: () => { newConversation(state, currentModel()?.id); state.view = 'chat'; persist(); renderShell(); renderView(); },
    }, h('span', { html: icon('plus', 15) }), h('span', { text: '新对话' })));
    const recent = h('div', { class: 'recent-list' });
    const conversations = recentConversations(state, 8);
    for (const conversation of conversations) {
      recent.append(h('div', { class: 'recent-row' },
        h('button', {
          class: `recent-item ${conversation.id === state.activeId ? 'active' : ''}`,
          type: 'button', title: conversation.title,
          onClick: () => {
            state.activeId = conversation.id;
            state.view = 'chat';
            persist();
            renderShell();
            renderView();
          },
        }, h('span', { html: icon('chat', 13) }), h('span', { text: conversation.title })),
        h('button', {
          class: 'recent-delete', type: 'button', title: '删除会话', html: icon('trash', 12),
          onClick: async (event) => {
            event.stopPropagation();
            const ok = await confirmDialog({
              title: '删除这个会话？', message: conversation.title,
              confirmLabel: '删除', danger: true,
            });
            if (!ok) return;
            const wasActive = state.activeId === conversation.id;
            deleteConversation(state, conversation.id);
            for (const message of conversation.messages) {
              if (message.mediaKey) void deleteMedia(message.mediaKey).catch(() => {});
            }
            if (!state.activeId && state.conversations.length) {
              state.activeId = recentConversations(state, 1)[0]?.id ?? null;
            }
            if (wasActive) { renderShell(); renderView(); } else { renderSidebarOnly(); }
            toast('会话已删除', 'success', 1400);
          },
        }),
      ));
    }
    if (!conversations.length) recent.append(h('div', { class: 'faint', style: 'padding:6px 9px;font-size:11.5px', text: '还没有对话' }));
    context.push(h('div', { class: 'sidebar-section' },
      h('div', { class: 'sidebar-heading' }, h('span', { text: '最近' }), h('span', { text: String(state.conversations.length) })),
      recent,
    ));
  } else if (state.view === 'media') {
    const nav = h('div', { class: 'sidebar-section' });
    nav.append(h('div', { class: 'sidebar-heading' }, h('span', { text: '媒体工具' })));
    for (const tool of mediaTools) {
      const count = tool.id === 'transcribe'
        ? models.filter(isAudioModel).length
        : tool.id === 'images' ? models.filter(isImageModel).length : 0;
      nav.append(h('button', {
        class: `nav-item ${state.mediaTab === tool.id ? 'active' : ''}`, type: 'button',
        onClick: () => openMedia(tool.id),
      },
        h('span', { html: icon(tool.icon, 16) }),
        h('span', { text: tool.label }),
        tool.planned
          ? h('span', { class: 'nav-badge', text: '规划' })
          : count ? h('span', { class: 'filter-count', text: String(count) }) : null,
      ));
    }
    context.push(nav);
  } else if (state.view === 'models') {
    const links = [
      ['online', '在线模型', 'cloud'],
      ['downloads', '下载', 'download'],
      ['installed', '已安装', 'models'],
      ['import', '导入', 'folder'],
      ['devices', '设备', 'cpu'],
    ];
    const nav = h('div', { class: 'sidebar-section' });
    nav.append(h('div', { class: 'sidebar-heading' }, h('span', { text: '模型分类' })));
    for (const [id, label, iconName] of links) {
      const count = id === 'downloads' ? activeDownloads().length : 0;
      nav.append(h('button', {
        class: `nav-item ${state.subview === id ? 'active' : ''}`, type: 'button',
        onClick: () => setSubview(id),
      }, h('span', { html: icon(iconName, 16) }), h('span', { text: label }),
        count ? h('span', { class: 'filter-count', text: String(count) }) : null));
    }
    context.push(nav);
  }
  const loaded = models.filter((model) => model.loaded).length;
  const deviceClasses = [...new Set(devices.map((device) => device.class))];
  context.push(h('div', { class: 'sidebar-footer' },
    h('div', { class: 'runtime-card' },
      h('div', { class: 'runtime-row' }, h('span', { class: 'status-dot' }), h('strong', { text: '本地运行时' })),
      h('div', { class: 'runtime-row runtime-detail' }, h('span', { text: `${loaded} 已加载 · ${models.length} 已配置` })),
      h('div', { class: 'runtime-row runtime-detail' }, h('span', { html: icon('cpu', 12) }), h('span', { text: deviceClasses.join(' / ') || '未检测设备' })),
    ),
  ));
  return h('aside', { class: 'sidebar' }, ...context);
}

function titlebar() {
  const tabs = h('nav', { class: 'top-tabs', attrs: { 'aria-label': '主导航' } });
  for (const [id, label, iconName] of navItems) {
    tabs.append(h('button', {
      class: `tab ${state.view === id ? 'active' : ''}`,
      type: 'button', onClick: () => navigate(id),
    }, h('span', { html: icon(iconName, 15) }), h('span', { text: label })));
  }
  const actions = h('div', { class: 'titlebar-actions' });
  actions.append(h('button', {
    class: 'btn btn-icon ghost', type: 'button', title: `主题：${themeLabel(state.theme)}`,
    attrs: { 'data-theme-toggle': '' }, onClick: () => cycleTheme(),
    html: icon(themeIcon(state.theme), 16),
  }));
  const controls = h('div', { class: 'window-controls' },
    h('button', { class: 'window-btn', type: 'button', title: '最小化', onClick: () => windowCommand('window_minimize'), html: icon('minus', 14) }),
    h('button', { class: 'window-btn', type: 'button', title: '最大化', onClick: () => windowCommand('window_toggle_maximize'), html: icon('square', 12) }),
    h('button', { class: 'window-btn close', type: 'button', title: '关闭', onClick: () => windowCommand('window_close'), html: icon('x', 14) }),
  );
  return h('header', {
    class: 'titlebar',
    attrs: { 'data-tauri-drag-region': '' },
    onMousedown: (event) => {
      if (event.button !== 0) return;
      if (event.target.closest('button, select, input, a, .window-controls')) return;
      windowCommand('window_start_dragging');
    },
    onDblclick: (event) => {
      if (event.target.closest('button, select, input, a, .window-controls')) return;
      windowCommand('window_toggle_maximize');
    },
  },
    h('div', { class: 'topbar-brand', attrs: { 'data-tauri-drag-region': '' } },
      h('div', { class: 'brand-mark', html: icon('sparkles', 15) }),
      h('div', { class: 'brand-text' }, h('strong', { text: 'UniInfer' }), h('span', { text: '本地统一推理' })),
    ),
    tabs,
    actions,
    controls,
  );
}

function windowCommand(command) {
  window.__TAURI__?.core?.invoke(command).catch?.(() => {});
}

function themeLabel(theme) {
  return theme === 'light' ? '浅色' : theme === 'dark' ? '深色' : '跟随系统';
}

function renderShell() {
  chooseModelForView(state.view);
  app.dataset.view = state.view;
  const main = h('main', { class: 'main', id: 'view' });
  app.replaceChildren(titlebar(), sidebar(), h('div', { class: 'workspace' }, main));
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
    mediaTab,
    setMediaTab: (tab) => openMedia(tab),
    openMedia,
    subview: () => state.subview,
    setSubview,
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

function openMedia(tab) {
  if (!['transcribe', 'images', 'tts', 'video'].includes(tab)) return;
  state.mediaTab = tab;
  state.view = 'media';
  persist();
  chooseModelForView('media');
  renderShell();
  renderView();
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
  if (view === 'media') state.mediaTab = mediaTab();
  if (view === 'models' && !['online', 'downloads', 'installed', 'devices', 'import'].includes(state.subview)) state.subview = 'online';
  persist();
  chooseModelForView(view);
  renderShell();
  renderView();
}

function setSubview(subview) {
  state.subview = subview;
  persist();
  renderShell();
  renderView();
}

function watchDownloads() {
  const previous = new Map();
  let activeCount = -1;
  subscribeDownloads((tasks) => {
    const active = tasks.filter((task) => !['done', 'failed', 'cancelled'].includes(task.status));
    if (active.length !== activeCount) {
      activeCount = active.length;
      renderSidebarOnly();
    }
    for (const task of tasks) {
      const before = previous.get(task.id);
      previous.set(task.id, task.status);
      if (before && before !== 'done' && task.status === 'done') {
        toast(`${task.repo} 下载完成并已加入模型库`, 'success');
        void refreshModels({ silent: true }).catch(() => {});
      }
    }
  });
}

async function refreshModels({ silent = false } = {}) {
  try {
    const [modelBody, deviceBody] = await Promise.all([api.models(), api.devices()]);
    models = sortModels((modelBody.data ?? []).filter((model) => !model.id.startsWith('tiny-')));
    devices = deviceBody.devices ?? [];
    chooseModelForView(state.view);
    if (silent) {
      renderSidebarOnly();
      document.dispatchEvent(new CustomEvent('uniinfer:model'));
    } else {
      renderShell();
      renderView();
    }
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
  await initDownloads();
  watchDownloads();
  try {
    const [modelBody, deviceBody] = await Promise.all([api.models(), api.devices()]);
    models = sortModels((modelBody.data ?? []).filter((model) => !model.id.startsWith('tiny-')));
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
document.addEventListener('dragover', (event) => event.preventDefault());
document.addEventListener('drop', (event) => event.preventDefault());

void boot();
