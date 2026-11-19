import { h, clear } from '../dom.js';
import { icon } from '../icons.js';
import { errorMessage, formatBytes, isAudioModel, downloadText } from '../format.js';
import { toast, emptyState, copyText } from '../components/ui.js';

export function mountTranscribe(root, ctx) {
  const state = ctx.state;
  const saved = state.ui.transcribe.result;
  let file = null;
  let busy = false;
  let result = saved ? { text: saved.text, duration: saved.duration } : null;

  const modelSelect = h('select', { class: 'select', attrs: { 'aria-label': '转写模型' } });
  function syncModels() {
    const list = ctx.models().filter((model) => isAudioModel(model) && !model.route_error);
    const current = ctx.currentModel()?.id ?? modelSelect.value;
    clear(modelSelect);
    for (const model of list) modelSelect.append(h('option', { value: model.id, text: model.id }));
    if (!list.length) return;
    const preferred = list.find((model) => model.id === current)
      ?? list.find((model) => model.id === saved?.model)
      ?? list[0];
    modelSelect.value = preferred.id;
  }
  syncModels();
  modelSelect.addEventListener('change', () => {
    ctx.setModel(modelSelect.value);
    ctx.persistSoon?.();
  });

  const fileInput = h('input', {
    type: 'file', accept: '.wav,audio/wav,audio/x-wav', class: 'visually-hidden',
    onChange: (event) => setFile(event.target.files?.[0] ?? null),
  });
  const dropzone = h('div', { class: 'dropzone', role: 'button', tabIndex: 0 },
    h('div', { html: icon('wave', 26) }),
    h('strong', { text: '拖入 WAV 音频，或点击选择文件' }),
    h('span', { class: 'faint', text: '16 kHz 单声道效果最佳 · 支持 WAV' }),
  );
  const fileSlot = h('div');
  const transcribeButton = h('button', { class: 'btn primary lg', type: 'button', disabled: true, onClick: () => void run() },
    h('span', { html: icon('play', 15) }), h('span', { text: '开始转写' }));
  const status = h('span', { class: 'faint' });
  const resultText = h('div', { class: 'result-box', text: saved?.text || '转写结果会显示在这里。' });
  const meta = h('div', { class: 'row faint result-meta' });
  const output = h('div', { class: 'card result-card' },
    h('div', { class: 'section-title' }, h('h2', { text: '结果' }), h('span', { class: 'spacer' }),
      h('button', { class: 'btn sm ghost', type: 'button', disabled: true, onClick: () => copyText(result?.text ?? ''), html: `${icon('copy', 14)} 复制` }),
      h('button', { class: 'btn sm ghost', type: 'button', disabled: true, onClick: () => downloadText('transcript.txt', result?.text ?? ''), html: `${icon('download', 14)} 下载` }),
    ),
    resultText, meta,
  );
  const copyButton = output.querySelectorAll('button')[0];
  const downloadButton = output.querySelectorAll('button')[1];

  function renderResult(metaItems = []) {
    if (!result?.text) {
      resultText.className = 'result-box';
      resultText.textContent = '转写结果会显示在这里。';
      meta.replaceChildren();
      copyButton.disabled = true;
      downloadButton.disabled = true;
      return;
    }
    resultText.className = 'result-box';
    resultText.textContent = result.text;
    meta.replaceChildren(...metaItems);
    copyButton.disabled = false;
    downloadButton.disabled = false;
  }

  function restoreResult() {
    if (!saved?.text) return;
    const metaItems = [];
    if (saved.model) metaItems.push(h('span', { class: 'chip', text: saved.model }));
    if (saved.duration) metaItems.push(h('span', { class: 'chip', text: `音频 ${Number(saved.duration).toFixed(1)} 秒` }));
    if (saved.fileName) metaItems.push(h('span', { class: 'chip', text: saved.fileName }));
    renderResult(metaItems);
  }

  function setFile(next) {
    if (!next) return;
    if (!next.name.toLowerCase().endsWith('.wav')) {
      toast('当前只支持 WAV 文件', 'error');
      fileInput.value = '';
      return;
    }
    if (next.size > 128 * 1024 * 1024) {
      toast('音频文件超过 128 MiB', 'error');
      return;
    }
    file = next;
    fileInput.value = '';
    fileSlot.replaceChildren(h('div', { class: 'file-card' },
      h('span', { class: 'icon', html: icon('wave', 18) }),
      h('div', { style: 'min-width:0;flex:1' },
        h('div', { class: 'file-name', text: next.name }),
        h('div', { class: 'file-meta', text: `${formatBytes(next.size)} · 已就绪` }),
      ),
      h('button', { class: 'btn btn-icon ghost sm', type: 'button', title: '移除', onClick: () => { file = null; fileSlot.replaceChildren(); transcribeButton.disabled = true; status.textContent = ''; }, html: icon('x', 13) }),
    ));
    transcribeButton.disabled = false;
    status.textContent = '';
  }

  async function run() {
    if (!file || busy) return;
    const model = modelSelect.value;
    if (!model) { toast('没有可用的转写模型', 'error'); return; }
    busy = true;
    transcribeButton.disabled = true;
    transcribeButton.textContent = '转写中…';
    status.textContent = '正在加载模型并识别音频…';
    try {
      result = await ctx.api.transcribe(file, model);
      const duration = Number(result.duration ?? 0);
      state.ui.transcribe.result = {
        text: result.text || '（空结果）',
        duration,
        model,
        fileName: file.name,
      };
      ctx.persist();
      renderResult([
        h('span', { class: 'chip', text: model }),
        duration ? h('span', { class: 'chip', text: `音频 ${duration.toFixed(1)} 秒` }) : null,
        h('span', { class: 'chip', text: 'WAV' }),
      ]);
      status.textContent = '完成';
      toast('转写完成', 'success');
    } catch (error) {
      const message = errorMessage(error);
      resultText.textContent = message;
      resultText.classList.add('notice', 'error');
      status.textContent = '失败';
      toast(message, 'error');
    } finally {
      busy = false;
      transcribeButton.disabled = !file;
      transcribeButton.textContent = '开始转写';
    }
  }

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
  });
  dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragging');
    setFile(event.dataTransfer.files?.[0] ?? null);
  });

  if (!modelSelect.options.length) {
    root.append(h('div', { class: 'page' },
      emptyState({
        iconName: 'wave',
        title: '没有可用的转写模型',
        description: '在模型页导入 Whisper ONNX 或 GGML 模型后，这里会自动出现。',
        action: h('button', { class: 'btn', type: 'button', text: '打开模型页', onClick: () => ctx.navigate('models') }),
      }),
    ));
    const reloadWhenReady = () => {
      if (ctx.models().some(isAudioModel)) ctx.reload?.();
    };
    document.addEventListener('uniinfer:model', reloadWhenReady);
    return () => document.removeEventListener('uniinfer:model', reloadWhenReady);
  }

  restoreResult();
  const onModels = () => syncModels();
  document.addEventListener('uniinfer:model', onModels);
  root.append(h('div', { class: 'transcribe-view' }, h('div', { class: 'page' },
    h('div', { class: 'page-head' },
      h('h2', { text: '语音转写' }),
      h('p', { text: '把会议录音、访谈或语音备忘转成可编辑文本。模型在本机运行。' }),
    ),
    h('div', { class: 'split' },
      h('div', { class: 'card', style: 'display:grid;gap:14px' },
        h('div', { class: 'field' }, h('label', { text: '转写模型' }), modelSelect),
        dropzone,
        fileSlot,
        h('div', { class: 'toolbar' }, transcribeButton, status),
        fileInput,
      ),
      output,
    ),
  )));
  return () => document.removeEventListener('uniinfer:model', onModels);
}
