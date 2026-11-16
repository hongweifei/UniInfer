import { describe, expect, test } from 'bun:test';
import { modelCategories, modelTags, onlineCategories, unsupportedUseReason } from './model-meta.js';

const chat = { inputs: ['text'], outputs: ['text'], format: 'onnx', quantization: 'int4', engine: 'onnx' };
const vision = { inputs: ['text', 'image'], outputs: ['text'], format: 'gguf', engine: 'llama' };
const audio = { inputs: ['audio'], outputs: ['text'], format: 'ggml', engine: 'whisper' };
const image = { inputs: ['text'], outputs: ['image'], format: 'onnx', engine: 'onnx' };

describe('model metadata', () => {
  test('classifies installed models by capability', () => {
    expect(modelCategories(chat)).toContain('chat');
    expect(modelCategories(vision)).toContain('vision');
    expect(modelCategories(audio)).toContain('audio');
    expect(modelCategories(image)).toContain('image');
  });

  test('deduplicates tags', () => {
    expect(modelTags(chat)).toEqual(expect.arrayContaining(['ONNX', 'int4', '文本']));
    expect(modelTags(audio)).toContain('转写');
  });

  test('marks unimplemented output backends', () => {
    expect(unsupportedUseReason({ outputs: ['video'] })).toContain('视频');
    expect(unsupportedUseReason({ outputs: ['audio'] })).toContain('语音合成');
    expect(unsupportedUseReason({ outputs: ['text'] })).toBeNull();
  });

  test('classifies online tasks without confusing video diffusion with image', () => {
    expect(onlineCategories({ task: 'text-to-video', tags: ['diffusion'] })).toContain('video');
    expect(onlineCategories({ task: 'text-to-video', tags: ['diffusion'] })).not.toContain('image');
    expect(onlineCategories({ task: 'text-to-image', tags: ['diffusion'] })).toContain('image');
  });
});
