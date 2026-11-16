import { h, clear } from '../dom.js';
import { icon } from '../icons.js';

export function toast(message, type = 'info', timeout = 3200) {
  const region = document.getElementById('toast-region');
  const node = h('div', { class: `toast ${type}`, text: String(message) });
  region.append(node);
  window.setTimeout(() => node.remove(), timeout);
  return node;
}

export function openModal({ title, body, actions = [], width }) {
  const root = document.getElementById('modal-root');
  const close = () => {
    backdrop.remove();
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (event) => {
    if (event.key === 'Escape') close();
  };
  const foot = h('div', { class: 'modal-foot' });
  for (const action of actions) {
    foot.append(h('button', {
      class: `btn ${action.kind ?? ''}`,
      type: 'button',
      onClick: () => {
        Promise.resolve(action.onClick?.()).then((result) => {
          if (action.close !== false && result !== false) close();
        });
      },
      text: action.label,
    }));
  }
  const modal = h('div', { class: 'modal', style: width ? `width:${width}` : null },
    h('div', { class: 'modal-head' },
      h('h2', { text: title }),
      h('button', { class: 'btn btn-icon ghost', type: 'button', onClick: close, 'aria-label': '关闭', html: icon('x', 16) }),
    ),
    h('div', { class: 'modal-body' }, body),
    actions.length ? foot : null,
  );
  const backdrop = h('div', { class: 'modal-backdrop', onClick: (event) => { if (event.target === backdrop) close(); } }, modal);
  root.append(backdrop);
  document.addEventListener('keydown', onKey);
  requestAnimationFrame(() => modal.querySelector('button, input, textarea, select')?.focus());
  return { close, modal };
}

export function confirmDialog({ title, message, confirmLabel = '确认', danger = false }) {
  return new Promise((resolve) => {
    const { close } = openModal({
      title,
      body: h('p', { class: 'muted', text: message }),
      actions: [
        { label: '取消', kind: 'ghost', onClick: () => { resolve(false); close(); } },
        { label: confirmLabel, kind: danger ? 'danger' : 'primary', onClick: () => { resolve(true); close(); } },
      ],
    });
  });
}

export function emptyState({ iconName = 'sparkles', title, description, action }) {
  return h('div', { class: 'empty-state' },
    h('div', { class: 'empty-icon', html: icon(iconName, 25) }),
    h('h2', { text: title }),
    description ? h('p', { text: description }) : null,
    action ?? null,
  );
}

export function chip(text, kind = '') {
  return h('span', { class: `chip ${kind}`, text });
}

export function setBusy(button, busy, label) {
  button.disabled = busy;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label ?? '处理中…';
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
    delete button.dataset.label;
  }
}

export async function copyText(text) {
  await navigator.clipboard.writeText(text);
  toast('已复制', 'success', 1400);
}

export function renderText(container, text) {
  clear(container);
  const source = String(text ?? '');
  const pattern = /```([^\n]*)\n([\s\S]*?)```/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(source)) !== null) {
    if (match.index > cursor) container.append(document.createTextNode(source.slice(cursor, match.index)));
    const code = h('pre', { attrs: { 'data-language': match[1].trim() || 'code' } });
    code.textContent = match[2].replace(/\n$/, '');
    container.append(code);
    cursor = pattern.lastIndex;
  }
  if (cursor < source.length) container.append(document.createTextNode(source.slice(cursor)));
  if (!source) container.append(h('span', { class: 'typing', html: '<i></i><i></i><i></i>' }));
}

export function themeIcon(theme) {
  return theme === 'light' ? 'sun' : theme === 'dark' ? 'moon' : 'monitor';
}
