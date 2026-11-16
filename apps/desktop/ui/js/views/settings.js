import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { toast, setBusy, confirmDialog, copyText } from '../components/ui.js';
import { errorMessage } from '../format.js';

export function mountSettings(root, ctx) {
  const { state } = ctx;
  let config = null;

  const themeSwitch = h('div', { class: 'theme-switch' });
  const maxTokens = h('input', {
    class: 'input', type: 'number', min: '1', max: '262144', value: String(state.chatOptions.maxTokens),
    onChange: (event) => {
      state.chatOptions.maxTokens = Math.max(1, Number(event.target.value) || 512);
      event.target.value = state.chatOptions.maxTokens;
      ctx.persist();
      toast('已保存对话默认值', 'success');
    },
  });
  const temperature = h('input', {
    class: 'input', type: 'number', min: '0', max: '2', step: '0.1', value: String(state.chatOptions.temperature),
    onChange: (event) => {
      state.chatOptions.temperature = Math.min(2, Math.max(0, Number(event.target.value) || 0));
      event.target.value = state.chatOptions.temperature;
      ctx.persist();
      toast('已保存对话默认值', 'success');
    },
  });

  function renderTheme() {
    clear(themeSwitch);
    for (const [value, label, iconName] of [['system', '跟随系统', 'monitor'], ['light', '浅色', 'sun'], ['dark', '深色', 'moon']]) {
      themeSwitch.append(h('button', {
        type: 'button', class: state.theme === value ? 'active' : '',
        onClick: () => { ctx.setTheme(value); renderTheme(); },
      }, h('span', { html: icon(iconName, 13) }), h('span', { text: label })));
    }
  }

  const maxLoaded = h('input', { class: 'input', type: 'number', min: '0', value: '0' });
  const requestTimeout = h('input', { class: 'input', type: 'number', min: '0', step: '1', value: '0' });
  const loadTimeout = h('input', { class: 'input', type: 'number', min: '0', step: '1', value: '0' });
  const queueTimeout = h('input', { class: 'input', type: 'number', min: '0', step: '1', value: '30' });
  const configPath = h('span', { class: 'mono muted', text: '读取中…' });
  const runtimeStatus = h('span', { class: 'faint' });
  const saveRuntimeButton = h('button', {
    class: 'btn sm primary', type: 'button', disabled: true,
    onClick: () => void saveRuntime(),
  }, h('span', { html: icon('check', 14) }), h('span', { text: '保存运行时设置' }));
  const apiCopy = h('button', {
    class: 'btn btn-icon ghost sm', type: 'button', title: '复制 API 地址',
    onClick: () => void copyText(ctx.api.base),
    html: icon('copy', 13),
  });
  const tokenCopy = h('button', {
    class: 'btn btn-icon ghost sm', type: 'button', title: '复制 API Token',
    onClick: () => void copyText(ctx.api.token),
    html: icon('copy', 13),
  });
  const restartButton = h('button', {
    class: 'btn', type: 'button',
    onClick: () => void confirmDialog({
      title: '重启 UniInfer？',
      message: '保存的模型和运行时设置会在重启后加载。当前对话会保留在本地。',
      confirmLabel: '重启',
    }).then((ok) => { if (ok) ctx.restart(); }),
  }, h('span', { html: icon('refresh', 14) }), h('span', { text: '重启应用' }));

  async function loadRuntime() {
    try {
      config = await ctx.api.config();
      const server = config.config?.server ?? {};
      maxLoaded.value = String(server.max_loaded_models ?? 0);
      requestTimeout.value = String(Math.round((server.request_timeout_ms ?? 0) / 1000));
      loadTimeout.value = String(Math.round((server.model_load_timeout_ms ?? 0) / 1000));
      queueTimeout.value = String(Math.round((server.queue_timeout_ms ?? 30000) / 1000));
      configPath.textContent = config.path ?? '未绑定配置文件';
      runtimeStatus.textContent = config.token_configured ? 'API token 已配置' : '';
      saveRuntimeButton.disabled = false;
    } catch (error) {
      runtimeStatus.textContent = errorMessage(error);
      toast(runtimeStatus.textContent, 'error');
    }
  }

  async function saveRuntime() {
    if (!config) return;
    setBusy(saveRuntimeButton, true, '保存中…');
    try {
      const body = {
        ...config.config,
        server: {
          ...(config.config.server ?? {}),
          max_loaded_models: Math.max(0, Number(maxLoaded.value) || 0),
          request_timeout_ms: Math.max(0, Number(requestTimeout.value) || 0) * 1000,
          model_load_timeout_ms: Math.max(0, Number(loadTimeout.value) || 0) * 1000,
          queue_timeout_ms: Math.max(0, Number(queueTimeout.value) || 0) * 1000,
        },
      };
      const result = await ctx.api.saveConfig(body);
      runtimeStatus.textContent = result.restart_required ? '已保存，重启后生效' : '已保存';
      toast(runtimeStatus.textContent, 'success');
      await loadRuntime();
    } catch (error) {
      runtimeStatus.textContent = errorMessage(error);
      toast(runtimeStatus.textContent, 'error');
    } finally {
      setBusy(saveRuntimeButton, false);
    }
  }

  renderTheme();
  void loadRuntime();

  root.append(h('div', { class: 'settings-view' }, h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h2', { text: '设置' }),
      h('p', { text: '外观、对话默认值、运行时与本地数据。' }),
    ),
    h('div', { class: 'settings-grid' },
      h('div', { class: 'settings-panel' },
        h('div', { class: 'card', attrs: { id: 'settings-appearance' } },
          h('div', { class: 'section-title' }, h('h2', { text: '外观' })),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '主题' }), h('span', { text: '选择界面明暗，或跟随系统。' })),
            themeSwitch,
          ),
        ),
        h('div', { class: 'card', attrs: { id: 'settings-chat' } },
          h('div', { class: 'section-title' }, h('h2', { text: '对话默认值' })),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '最大输出 tokens' }), h('span', { text: '每次回复的上限。' })),
            h('div', { style: 'width:140px' }, maxTokens),
          ),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '温度' }), h('span', { text: '0 更稳定，2 更发散。' })),
            h('div', { style: 'width:140px' }, temperature),
          ),
        ),
        h('div', { class: 'card', attrs: { id: 'settings-connection' } },
          h('div', { class: 'section-title' }, h('h2', { text: '本地连接' }), h('span', { class: 'spacer' }), h('span', { class: 'chip success', text: '运行中' })),
          h('div', { class: 'model-kv' },
            h('strong', { text: 'API' }), h('span', { class: 'row', style: 'gap:4px' }, h('span', { class: 'mono', text: ctx.api.base }), apiCopy),
            h('strong', { text: 'Token' }), h('span', { class: 'row', style: 'gap:4px' }, h('span', { class: 'mono muted', text: '已配置' }), tokenCopy),
            h('strong', { text: '模型' }), h('span', { text: `${ctx.models().length} 个已配置` }),
            h('strong', { text: '配置文件' }), configPath,
          ),
          h('div', { class: 'row', style: 'margin-top:12px' }, restartButton),
        ),
        h('div', { class: 'card', attrs: { id: 'settings-runtime' } },
          h('div', { class: 'section-title' }, h('h2', { text: '运行时' }), h('span', { class: 'spacer' }), runtimeStatus),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '最大驻留模型数' }), h('span', { text: '0 表示不限制；超出后自动卸载最久未用模型。' })),
            h('div', { style: 'width:140px' }, maxLoaded),
          ),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '请求超时' }), h('span', { text: '单位：秒；0 表示不设截止时间。' })),
            h('div', { style: 'width:140px' }, requestTimeout),
          ),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '模型加载超时' }), h('span', { text: '单位：秒；0 表示不限制加载时间。' })),
            h('div', { style: 'width:140px' }, loadTimeout),
          ),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '排队超时' }), h('span', { text: '单位：秒；忙碌时最多等待多久。' })),
            h('div', { style: 'width:140px' }, queueTimeout),
          ),
          h('div', { class: 'row', style: 'margin-top:12px' }, saveRuntimeButton),
        ),
        h('div', { class: 'card', attrs: { id: 'settings-data' } },
          h('div', { class: 'section-title' }, h('h2', { text: '本地数据' })),
          h('div', { class: 'setting-row' },
            h('div', { class: 'setting-label' }, h('strong', { text: '对话历史' }), h('span', { text: `保存在此设备，共 ${state.conversations.length} 个会话。` })),
            h('button', {
              class: 'btn sm danger', type: 'button',
              onClick: async () => {
                if (await confirmDialog({ title: '清除对话历史？', message: '这会删除本机保存的所有对话，无法恢复。', confirmLabel: '清除', danger: true })) {
                  ctx.clearHistory();
                  toast('对话历史已清除', 'success');
                }
              },
            }, h('span', { html: icon('trash', 14) }), h('span', { text: '清除' })),
          ),
        ),
      ),
    ),
  )));

  return () => {};
}
