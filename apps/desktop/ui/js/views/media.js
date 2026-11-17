import { h } from '../dom.js';
import { icon } from '../icons.js';
import { mountImages } from './images.js';
import { mountTranscribe } from './transcribe.js';

const tabs = [
  { id: 'transcribe', label: '语音转写', icon: 'wave', mount: mountTranscribe },
  { id: 'images', label: '图像生成', icon: 'image', mount: mountImages },
  { id: 'tts', label: '语音合成', icon: 'volume', planned: true, description: '文本转语音，输出可播放的音频。' },
  { id: 'video', label: '视频生成', icon: 'play', planned: true, description: '文本或图像生成视频，支持任务队列与取消。' },
];

export function mountMedia(root, ctx) {
  const activeId = tabs.some((tab) => tab.id === ctx.mediaTab()) ? ctx.mediaTab() : 'transcribe';
  const active = tabs.find((tab) => tab.id === activeId);
  const body = h('div', { class: 'media-body' });

  const nav = h('div', { class: 'media-nav', attrs: { 'aria-label': '媒体工具' } });
  for (const tab of tabs) {
    nav.append(h('button', {
      class: `media-tab ${tab.id === activeId ? 'active' : ''}`,
      type: 'button',
      onClick: () => ctx.setMediaTab(tab.id),
      title: tab.planned ? `${tab.label}：规划中` : tab.label,
    },
      h('span', { html: icon(tab.icon, 15) }),
      h('span', { text: tab.label }),
      tab.planned ? h('span', { class: 'media-soon', text: '规划中' }) : null,
    ));
  }

  root.append(h('div', { class: 'media-view' }, nav, body));

  if (active.planned) {
    body.append(h('div', { class: 'media-planned page' },
      h('div', { class: 'empty-state' },
        h('div', { class: 'empty-icon', html: icon(active.icon, 24) }),
        h('h2', { text: `${active.label}即将支持` }),
        h('p', { text: active.description }),
        h('button', {
          class: 'btn', type: 'button', text: '返回语音转写',
          onClick: () => ctx.setMediaTab('transcribe'),
        }),
      ),
    ));
    return () => {};
  }

  return active.mount(body, ctx) ?? (() => {});
}
