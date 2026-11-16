import { h } from '../dom.js';
import { icon } from '../icons.js';
import { timeAgo } from '../format.js';
import { toast, renderText, copyText } from './ui.js';

function speakText(text) {
  const synth = window.speechSynthesis;
  if (!synth) {
    toast('当前环境不支持语音朗读', 'error');
    return;
  }
  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(String(text ?? '').replace(/```[\s\S]*?```/g, '代码块'));
  utterance.lang = 'zh-CN';
  synth.speak(utterance);
}

export function messageNode(message) {
  const isUser = message.role === 'user';
  const text = h('div', { class: 'message-text' });
  renderText(text, message.content);
  const meta = h('div', { class: 'message-meta' },
    !isUser ? h('span', { text: 'UniInfer' }) : null,
    !message.pending ? h('span', { text: timeAgo(message.createdAt ?? Date.now()) }) : null,
  );
  const reasoningBody = h('div', { class: 'reasoning-body' });
  const reasoningSummary = h('span', { text: '思考过程' });
  const reasoning = isUser ? null : h('details', {
    class: 'message-reasoning',
    hidden: !message.reasoning,
  },
    h('summary', {}, h('span', { html: icon('sparkles', 13) }), reasoningSummary),
    reasoningBody,
  );
  if (reasoning && message.reasoning) renderText(reasoningBody, message.reasoning);
  const actions = h('div', { class: 'message-actions' });
  if (!message.pending) {
    if (!isUser && message.content) {
      actions.append(h('button', {
        class: 'btn btn-icon ghost', type: 'button', title: '系统朗读（本地 TTS 未接入）',
        onClick: () => speakText(message.content),
        html: icon('volume', 13),
      }));
    }
    actions.append(h('button', {
      class: 'btn btn-icon ghost', type: 'button', title: '复制',
      onClick: () => copyText(message.content ?? ''),
      html: icon('copy', 13),
    }));
  }
  const body = h('div', { class: 'message-body' }, meta, reasoning, text, actions);
  const node = h('div', { class: `message ${isUser ? 'user' : 'assistant'}${message.error ? ' error' : ''}` },
    h('div', { class: 'avatar', text: isUser ? '你' : 'U' }),
    body,
  );
  if (message.generated && message.images?.length) {
    const gallery = h('div', { class: 'generated-media' });
    for (const url of message.images) {
      gallery.append(h('img', { src: url, alt: message.content || '生成图片' }));
    }
    body.insertBefore(gallery, text);
  } else if (message.generated && message.mediaKey) {
    body.insertBefore(h('div', { class: 'notice faint', text: '生成图片缓存不可用' }), text);
  } else if (message.images?.length) {
    const gallery = h('div', { class: 'attachments' });
    for (const url of message.images) {
      gallery.append(h('div', { class: 'attachment' }, h('img', { src: url, alt: '图片附件' })));
    }
    body.insertBefore(gallery, text);
  } else if (message.imageCount) {
    body.insertBefore(h('div', { class: 'row' }, h('span', { class: 'chip', text: `${message.imageCount} 张图片` })), text);
  }
  return { node, text, reasoning, reasoningBody, reasoningSummary };
}

