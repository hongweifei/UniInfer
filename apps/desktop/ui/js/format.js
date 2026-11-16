const modalityNames = { text: '文本', audio: '音频', image: '图片', video: '视频' };
const formatNames = {
  onnx: 'ONNX', gguf: 'GGUF', safetensors: 'Safetensors', ggml: 'GGML',
};

export function modality(value) {
  return modalityNames[value] ?? value ?? '未知';
}

export function formatName(value) {
  return formatNames[value] ?? String(value ?? '').toUpperCase();
}

export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size >= 10 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

export function modelTitle(model) {
  return model?.id ?? '未选择模型';
}

export function modalities(list) {
  return (list ?? []).map(modality).join(' + ') || '无';
}

export function capabilitySummary(model) {
  const inputs = modalities(model?.inputs);
  const outputs = modalities(model?.outputs);
  return `${inputs} → ${outputs}`;
}

export function supports(model, input, output) {
  if (!model) return false;
  if (input && !(model.inputs ?? []).includes(input)) return false;
  if (output && !(model.outputs ?? []).includes(output)) return false;
  return true;
}

export function isAudioModel(model) {
  return supports(model, 'audio', 'text');
}

export function isImageModel(model) {
  return supports(model, null, 'image');
}

export function isChatModel(model) {
  return supports(model, 'text', 'text');
}

export function uid(prefix = 'id') {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

export function now() {
  return Date.now();
}

export function timeAgo(timestamp) {
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return '刚刚';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} 天前`;
  return new Date(timestamp).toLocaleDateString('zh-CN');
}

export function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error('读取文件失败'));
    reader.readAsDataURL(file);
  });
}

export function downloadText(filename, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  downloadBlob(filename, blob);
}

export function downloadDataURL(filename, dataUrl) {
  const anchor = document.createElement('a');
  anchor.href = dataUrl;
  anchor.download = filename;
  anchor.click();
}

export function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function errorMessage(error) {
  return error?.message ?? String(error ?? '未知错误');
}

export function friendlyRouteError(message) {
  const text = String(message ?? '');
  if (text.includes('no engine for runtime Whisper')) return '此模型需要 Whisper 引擎，当前构建未启用。';
  if (text.includes('no engine for runtime Llama')) return '此模型需要 llama.cpp 引擎，当前构建未启用。';
  if (text.includes('no engine for runtime Candle')) return '此模型需要 Candle 引擎，当前构建未启用。';
  if (text.includes('no engine implements')) return '当前构建没有能执行这个模型的引擎。';
  if (text.includes('unsupported modality')) return '当前引擎不支持这个模型需要的输入模态。';
  return text;
}


export const MODEL_CATEGORIES = [
  { id: 'chat', label: '对话' },
  { id: 'vision', label: '视觉' },
  { id: 'audio', label: '语音' },
  { id: 'image', label: '图像' },
  { id: 'video', label: '视频' },
  { id: 'embedding', label: '嵌入' },
  { id: 'other', label: '其他' },
];

function uniqueTags(tags) {
  const seen = new Map();
  for (const tag of tags) {
    const text = String(tag ?? '').trim();
    if (!text || text.toLowerCase() === 'unknown') continue;
    const key = text.toLowerCase();
    if (!seen.has(key)) seen.set(key, text);
  }
  return [...seen.values()];
}

export function modelCategories(model) {
  const categories = [];
  if (isChatModel(model)) categories.push('chat');
  if ((model?.inputs ?? []).includes('image')) categories.push('vision');
  if (isAudioModel(model) || (model?.outputs ?? []).includes('audio')) categories.push('audio');
  if (isImageModel(model)) categories.push('image');
  if ([...(model?.inputs ?? []), ...(model?.outputs ?? [])].includes('video')) categories.push('video');
  if (!categories.length) categories.push('other');
  return categories;
}

export function modelTags(model) {
  return uniqueTags([
    formatName(model?.format),
    model?.quantization,
    model?.engine,
    ...(model?.inputs ?? []).map((item) => modality(item)),
    ...(model?.outputs ?? []).map((item) => `→ ${modality(item)}`),
    model?.streaming_capable ? '流式' : null,
    isAudioModel(model) ? '转写' : null,
  ]);
}

export function onlineCategories(entry) {
  const text = `${entry?.task ?? ''} ${(entry?.tags ?? []).join(' ')}`.toLowerCase();
  const categories = [];
  if (/text-generation|conversational|instruction|chat|instruct/.test(text)) categories.push('chat');
  if (/image-text-to-text|vision|visual|multimodal|image-to-text/.test(text)) categories.push('vision');
  if (/automatic-speech-recognition|speech|asr|whisper|audio|text-to-speech|tts/.test(text)) categories.push('audio');
  if (/text-to-image|image-generation|image-to-image|stable-diffusion/.test(text)) categories.push('image');
  if (/text-to-video|video|image-to-video/.test(text)) categories.push('video');
  if (/feature-extraction|sentence-similarity|embedding|rerank|retrieval/.test(text)) categories.push('embedding');
  if (!categories.length) categories.push('other');
  return categories;
}

export function onlineTags(entry) {
  return uniqueTags([entry?.task, ...(entry?.tags ?? [])].filter(Boolean));
}
