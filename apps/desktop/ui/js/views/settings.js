import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { toast, setBusy, confirmDialog } from '../components/ui.js';
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

  const configPath = h('div', { class: 'mono muted', text: '读取中…' });
  const configEditor = h('textarea', { class: 'config-editor', spellcheck: false, attrs: { 'aria-label': '配置 JSON' } });
  const configStatus = h('span', { class: 'faint' });
  const loadConfigButton = h('button', { class: 'btn sm', type: 'button', onClick: () => void loadConfig() },
    h('span', { html: icon('refresh', 14) }), h('span', { text: '重新读取' }));
  const saveConfigButton = h('button', { class: 'btn sm primary', type: 'button', onClick: () => void saveConfig() },
    h('span', { html: icon('check', 14) }), h('span', { text: '保存配置' }));
  const restartButton = h('button', {
    class: 'btn', type: 'button',
    onClick: () => void confirmDialog({
      title: '重启 UniInfer？',
      message: '保存的模型和配置会在重启后加载。当前对话会保留在本地。',
      confirmLabel: '重启',
    }).then((ok) => { if (ok) ctx.restart(); }),
  }, h('span', { html: icon('refresh', 14) }), h('span', { text: '重启应用' }));

  async function loadConfig() {
    setBusy(loadConfigButton, true, '读取中…');
    try {
      config = await ctx.api.config();
      configPath.textContent = config.path ?? '未绑定配置文件';
      configEditor.value = JSON.stringify(config.config, null, 2);
      configStatus.textContent = config.token_configured ? 'token 已配置' : 'token 未配置';
    } catch (error) {
      configStatus.textContent = errorMessage(error);
      toast(configStatus.textContent, 'error');
    } finally {
      setBusy(loadConfigButton, false);
    }
  }

  async function saveConfig() {
    setBusy(saveConfigButton, true, '保存中…');
    try {
      const parsed = JSON.parse(configEditor.value);
      const result = await ctx.api.saveConfig(parsed);
      configStatus.textContent = result.restart_required ? '已保存，重启后生效' : '已保存';
      toast(configStatus.textContent, 'success');
      await loadConfig();
    } catch (error) {
      configStatus.textContent = errorMessage(error);
      toast(configStatus.textContent, 'error');
    } finally {
      setBusy(saveConfigButton, false);
    }
  }

  renderTheme();
  void loadConfig();

  root.append(h('div', { class: 'settings-view' }, h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h2', { text: '设置' }),
      h('p', { text: '外观、对话默认值和本地运行时。' }),
    ),
    h('div', { class: 'settings-grid' },
      h('nav', { class: 'settings-nav' },
        h('button', { class: 'active', type: 'button', text: '外观', onClick: () => scrollTo('settings-appearance') }),
        h('button', { type: 'button', text: '对话', onClick: () => scrollTo('settings-chat') }),
        h('button', { type: 'button', text: '连接', onClick: () => scrollTo('settings-connection') }),
        h('button', { type: 'button', text: '高级', onClick: () => scrollTo('settings-advanced') }),
      ),
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
          h('div', { class: 'section-title' }, h('h2', { text: '本地连接' }), h('span', { class: 'spacer' }), chipStatus()),
          h('div', { class: 'model-kv' },
            h('strong', { text: 'API' }), h('span', { class: 'mono', text: ctx.api.base }),
            h('strong', { text: '模型' }), h('span', { text: `${ctx.models().length} 个已配置` }),
            h('strong', { text: '配置文件' }), h('span', { class: 'mono', text: '见高级配置' }),
          ),
          h('div', { class: 'row', style: 'margin-top:12px' }, restartButton),
        ),
        h('div', { class: 'card', attrs: { id: 'settings-advanced' } },
          h('div', { class: 'section-title' }, h('h2', { text: '高级配置' }), h('span', { class: 'spacer' }), configStatus),
          h('p', { class: 'muted', style: 'margin-bottom:10px', text: '直接编辑 config JSON。保存后需要重启才能应用模型与服务器设置。' }),
          configPath,
          configEditor,
          h('div', { class: 'row', style: 'margin-top:10px' }, loadConfigButton, saveConfigButton),
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

  function scrollTo(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function chipStatus() {
    return h('span', { class: 'chip success', text: '运行中' });
  }
  return () => {};
}
