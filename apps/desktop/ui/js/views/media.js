import { h } from '../dom.js';
import { icon } from '../icons.js';
import { mountImages } from './images.js';
import { mountTranscribe } from './transcribe.js';

export const mediaTools = [
  { id: 'transcribe', label: '语音转写', icon: 'wave', mount: mountTranscribe },
  { id: 'images', label: '图像生成', icon: 'image', mount: mountImages },
  { id: 'tts', label: '语音合成', icon: 'volume', planned: true, description: '文本转语音，输出可播放的音频。' },
  { id: 'video', label: '视频生成', icon: 'play', planned: true, description: '文本或图像生成视频，支持任务队列与取消。' },
];

export function mountMedia(root, ctx) {
  const activeId = mediaTools.some((tab) => tab.id === ctx.mediaTab()) ? ctx.mediaTab() : 'transcribe';
  const active = mediaTools.find((tab) => tab.id === activeId);
  const body = h('div', { class: 'media-body' });

  root.append(h('div', { class: 'media-view' }, body));

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
