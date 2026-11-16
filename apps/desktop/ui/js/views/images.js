import { h } from '../dom.js';
import { icon } from '../icons.js';
import { errorMessage, isImageModel, downloadDataURL } from '../format.js';
import { toast, emptyState } from '../components/ui.js';

export function mountImages(root, ctx) {
  const imageModels = ctx.models().filter((model) => isImageModel(model) && !model.route_error);
  const state = ctx.state;
  if (!imageModels.length) {
    root.append(h('div', { class: 'page' },
      emptyState({
        iconName: 'image',
        title: '没有可用的图像模型',
        description: '当前运行时只包含图像生成 fixture。导入真实图像模型后，这里会显示可用的生成参数。',
        action: h('button', { class: 'btn', type: 'button', text: '打开模型页', onClick: () => ctx.navigate('models') }),
      }),
    ));
    return () => {};
  }

  const modelSelect = h('select', { class: 'select' });
  for (const model of imageModels) modelSelect.append(h('option', { value: model.id, text: model.id }));
  const preferred = imageModels.find((model) => model.id === state.selectedModel) ?? imageModels[0];
  if (preferred) modelSelect.value = preferred.id;

  const prompt = h('textarea', {
    class: 'textarea', rows: 4,
    placeholder: '描述你想生成的画面，例如：清晨的海边，薄雾，电影感光线',
  });
  const negative = h('input', { class: 'input', placeholder: '可选：不希望出现的元素' });
  const size = h('select', { class: 'select' },
    ['512x512', '768x768', '1024x1024', '64x64'].map((value) => h('option', { value, text: value })),
  );
  size.value = '512x512';
  const steps = h('input', { class: 'input', type: 'number', min: '1', max: '100', value: '20' });
  const guidance = h('input', { class: 'input', type: 'number', min: '0', max: '30', step: '0.5', value: '7' });
  const seed = h('input', { class: 'input', type: 'number', min: '0', placeholder: '随机' });
  const generateLabel = h('span', { text: '生成图片' });
  const generate = h('button', { class: 'btn primary lg', type: 'button', onClick: () => void run() },
    h('span', { html: icon('sparkles', 16) }), generateLabel);
  const status = h('span', { class: 'faint' });
  const preview = h('div', { class: 'preview' });
  const resultMeta = h('div', { class: 'row faint' });
  let current = null;

  function showEmpty() {
    preview.replaceChildren(h('div', { class: 'empty-state' },
      h('div', { class: 'empty-icon', html: icon('image', 24) }),
      h('h2', { text: '还没有生成图片' }),
      h('p', { text: '输入提示词后点击生成。真实扩散后端尚未接入，当前使用 fixture 验证管线。' }),
    ));
  }

  async function run() {
    const model = modelSelect.value;
    const text = prompt.value.trim();
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
      const response = await ctx.api.generateImage(body);
      const image = response.data?.[0];
      if (!image?.b64_json) throw new Error('响应中没有图片');
      current = image;
      preview.replaceChildren(h('img', { src: `data:${image.mime_type ?? 'image/png'};base64,${image.b64_json}`, alt: text }));
      resultMeta.replaceChildren(
        h('span', { class: 'chip', text: model }),
        h('span', { class: 'chip', text: `${image.width} × ${image.height}` }),
      );
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
    onClick: () => current && downloadDataURL('uniinfer-image.png', `data:${current.mime_type ?? 'image/png'};base64,${current.b64_json}`),
  }, h('span', { html: icon('download', 14) }), h('span', { text: '下载 PNG' }));

  const observer = new MutationObserver(() => { download.disabled = !current; });
  observer.observe(preview, { childList: true });

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
  showEmpty();
  return () => observer.disconnect();
}
