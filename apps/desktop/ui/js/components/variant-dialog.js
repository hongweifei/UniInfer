import { h } from '../dom.js';
import { formatBytes, formatName } from '../format.js';
import { chip, openModal } from './ui.js';

export function openVariantDialog({ entry, plan, onInstall }) {
  let selectedKey = plan.default_variant ?? plan.variants[0]?.key;
  const body = h('div', { class: 'variant-list' },
    h('p', { class: 'faint', text: '同一仓库包含多个格式或量化文件。选择要下载的变体；推荐项已预选。' }),
    ...plan.variants.map((variant) => {
      const input = h('input', {
        type: 'radio',
        name: `variant-${entry.id}`,
        value: variant.key,
        checked: variant.key === selectedKey,
        onChange: () => {
          selectedKey = variant.key;
          for (const node of body.querySelectorAll('.variant-option')) {
            node.classList.toggle('selected', node.contains(input));
          }
        },
      });
      return h('label', {
        class: `variant-option ${variant.key === selectedKey ? 'selected' : ''}`,
      },
      input,
      h('div', { class: 'variant-option-main' },
        h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap' },
          h('strong', { text: variant.label }),
          variant.recommended ? chip('推荐', 'success') : null,
          chip(formatName(variant.format)),
          variant.quantization && variant.quantization !== 'unknown' ? chip(variant.quantization) : null,
        ),
        h('div', { class: 'faint', text: `${variant.files} 个文件 · ${formatBytes(variant.total)}` }),
        variant.source ? h('div', { class: 'variant-source mono faint', text: variant.source }) : null,
      ));
    }),
  );
  let dialog;
  dialog = openModal({
    title: `选择下载变体 · ${entry.id}`,
    body,
    actions: [
      { label: '取消', kind: 'ghost' },
      {
        label: '开始下载', kind: 'primary',
        onClick: () => {
          dialog.close();
          onInstall(selectedKey);
          return false;
        },
      },
    ],
  });
}
