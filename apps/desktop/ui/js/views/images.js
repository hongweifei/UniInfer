import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { errorMessage, isImageModel, downloadDataURL } from '../format.js';
import { toast } from '../components/ui.js';
import { getMedia, putMedia } from '../media-store.js';

export function mountImages(root, ctx) {
  const state = ctx.state;
  const draft = state.ui.images;
  const modelSelect = h('select', { class: 'select', attrs: { 'aria-label': '图像生成模型' } });
  const prompt = h('textarea', {
    class: 'textarea', rows: 4, value: draft.prompt ?? '',
    placeholder: '描述你想生成的画面，例如：清晨的海边，薄雾，电影感光线',
  });
  const negative = h('input', { class: 'input', value: draft.negative ?? '', placeholder: '可选：不希望出现的元素' });
  const size = h('select', { class: 'select' },
    ['512x512', '768x768', '1024x1024', '64x64'].map((value) => h('option', { value, text: value })),
  );
  size.value = ['512x512', '768x768', '1024x1024', '64x64'].includes(draft.size) ? draft.size : '512x512';
  const steps = h('input', { class: 'input', type: 'number', min: '1', max: '100', value: String(draft.steps ?? 20) });
  const guidance = h('input', { class: 'input', type: 'number', min: '0', max: '30', step: '0.5', value: String(draft.guidance ?? 7) });
  const seed = h('input', { class: 'input', type: 'number', min: '0', value: draft.seed ?? '', placeholder: '随机' });

  function syncModels() {
    const list = ctx.models().filter((model) => isImageModel(model) && !model.route_error);
    const current = ctx.currentModel()?.id ?? modelSelect.value;
    clear(modelSelect);
    modelSelect.disabled = !list.length;
    if (!list.length) {
      modelSelect.append(h('option', { value: '', text: '选择模型' }));
      return;
    }
    for (const model of list) modelSelect.append(h('option', { value: model.id, text: model.id }));
    const preferred = list.find((model) => model.id === current)
      ?? list.find((model) => model.id === draft.last?.model)
      ?? list[0];
    modelSelect.value = preferred.id;
  }
  syncModels();
  modelSelect.addEventListener('change', () => {
    ctx.setModel(modelSelect.value);
  });

  function rememberDraft() {
    Object.assign(draft, {
      prompt: prompt.value,
      negative: negative.value,
      size: size.value,
      steps: Number(steps.value) || 1,
      guidance: Number(guidance.value) || 0,
      seed: seed.value,
    });
    ctx.persistSoon?.();
  }
  for (const field of [prompt, negative, steps, guidance, seed]) field.addEventListener('input', rememberDraft);
  size.addEventListener('change', rememberDraft);
  const generateLabel = h('span', { text: '生成图片' });
  const generate = h('button', { class: 'btn primary lg', type: 'button', onClick: () => void run() },
    h('span', { html: icon('sparkles', 16) }), generateLabel);
  const status = h('span', { class: 'faint' });
  const preview = h('div', { class: 'preview' });
  const resultMeta = h('div', { class: 'row faint' });
  let current = null;

  function showImage(url, meta = {}) {
    current = { url, mime: meta.mime ?? 'image/png' };
    preview.replaceChildren(h('img', { src: url, alt: meta.alt ?? '生成的图片' }));
    const chips = [];
    if (meta.model) chips.push(h('span', { class: 'chip', text: meta.model }));
    if (meta.width) chips.push(h('span', { class: 'chip', text: `${meta.width} × ${meta.height}` }));
    if (meta.label) chips.push(h('span', { class: 'chip', text: meta.label }));
    resultMeta.replaceChildren(...chips);
    download.disabled = false;
  }

  async function restoreLast() {
    if (!draft.last) { showEmpty(); return; }
    try {
      const url = await getMedia('images:last');
      if (!url) { showEmpty(); return; }
      showImage(url, { ...draft.last, alt: draft.last.prompt ?? '上次生成的图片' });
      status.textContent = '已恢复上次结果';
    } catch {
      showEmpty();
    }
  }

  function showEmpty() {
    preview.replaceChildren(h('div', { class: 'empty-state' },
      h('div', { class: 'empty-icon', html: icon('image', 24) }),
      h('h2', { text: '还没有生成图片' }),
      h('p', { text: '输入提示词后点击生成，图片会显示在这里。' }),
    ));
  }

  async function run() {
    const model = modelSelect.value;
    rememberDraft();
    const text = prompt.value.trim();
    if (!model) { toast('当前没有可用的图像模型', 'error'); return; }
    if (!text) { toast('请先输入提示词', 'error'); prompt.focus(); return; }
    generate.disabled = true;
    generateLabel.textContent = '生成中…';
    status.textContent = '正在生成…';
    try {
      const body = {
        model,
        prompt: text,
        n: 1,
        size: size.value,
        response_format: 'b64_json',
        steps: Number(steps.value) || 1,
        guidance: Number(guidance.value) || 0,
      };
      if (negative.value.trim()) body.negative_prompt = negative.value.trim();
      if (seed.value !== '') body.seed = Number(seed.value);
      const response = await ctx.api.generateImageStream(body, {
        onEvent: (event) => {
          if (event.type === 'progress') {
            status.textContent = `生成中… ${event.step}/${event.total}`;
            return;
          }
          if (event.type !== 'preview' || !event.image?.b64_json) return;
          const url = `data:${event.image.mime_type ?? 'image/png'};base64,${event.image.b64_json}`;
          preview.replaceChildren(h('img', { src: url, alt: `第 ${event.step} 步预览` }));
          resultMeta.replaceChildren(h('span', { class: 'chip', text: `第 ${event.step}/${event.total} 步` }));
          status.textContent = `生成中… ${event.step}/${event.total}`;
        },
      });
      const image = response.data?.[0];
      if (!image?.b64_json) throw new Error('响应中没有图片');
      const mime = image.mime_type ?? 'image/png';
      const url = `data:${mime};base64,${image.b64_json}`;
      draft.last = {
        model,
        prompt: text,
        width: image.width,
        height: image.height,
        mime,
      };
      ctx.persist();
      void putMedia('images:last', url).catch(() => {});
      showImage(url, { model, width: image.width, height: image.height, alt: text });
      status.textContent = '完成';
      toast('图片已生成', 'success');
    } catch (error) {
      const message = errorMessage(error);
      status.textContent = '失败';
      toast(message, 'error');
      preview.replaceChildren(h('div', { class: 'empty-state' }, h('p', { class: 'notice error', text: message })));
    } finally {
      generate.disabled = false;
      generateLabel.textContent = '生成图片';
    }
  }

  const download = h('button', {
    class: 'btn sm', type: 'button', disabled: true,
    onClick: () => current && downloadDataURL('uniinfer-image.png', current.url),
  }, h('span', { html: icon('download', 14) }), h('span', { text: '下载 PNG' }));

  const observer = new MutationObserver(() => { download.disabled = !current; });
  observer.observe(preview, { childList: true });
  const onModels = () => syncModels();
  document.addEventListener('uniinfer:model', onModels);

  root.append(h('div', { class: 'images-view' }, h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h2', { text: '图像生成' }),
      h('p', { text: '用文字描述画面。参数保持简单，默认值适合大多数模型。' }),
    ),
    h('div', { class: 'image-layout' },
      h('div', { class: 'card', style: 'display:grid;gap:14px' },
        h('div', { class: 'field' }, h('label', { text: '模型' }), modelSelect),
        h('div', { class: 'field' }, h('label', { text: '提示词' }), prompt),
        h('div', { class: 'field' }, h('label', { text: '负面提示词' }), negative),
        h('div', { class: 'param-grid' },
          h('div', { class: 'field' }, h('label', { text: '尺寸' }), size),
          h('div', { class: 'field' }, h('label', { text: '步数' }), steps),
          h('div', { class: 'field' }, h('label', { text: '引导强度' }), guidance),
          h('div', { class: 'field' }, h('label', { text: '随机种子' }), seed),
        ),
        h('div', { class: 'toolbar' }, generate, status),
      ),
      h('div', { class: 'card', style: 'display:grid;gap:12px' },
        preview, resultMeta, h('div', { class: 'row' }, download),
      ),
    ),
  )));
  void restoreLast();
  return () => {
    observer.disconnect();
    document.removeEventListener('uniinfer:model', onModels);
  };
}
