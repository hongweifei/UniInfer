import { formatName, modality, isAudioModel, isImageModel, isChatModel } from './format.js';


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
