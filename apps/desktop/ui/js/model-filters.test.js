import { describe, expect, test } from 'bun:test';
import { filterModels, modelTagOptions } from './model-filters.js';

const models = [
  { id: 'chat-onnx', inputs: ['text'], outputs: ['text'], format: 'onnx', quantization: 'int4', loaded: true },
  { id: 'vision-gguf', inputs: ['text', 'image'], outputs: ['text'], format: 'gguf', loaded: false, route_error: null },
  { id: 'audio-ggml', inputs: ['audio'], outputs: ['text'], format: 'ggml', loaded: false, route_error: { message: 'unsupported' } },
];

describe('installed model filters', () => {
  test('filters by category and tag', () => {
    expect(filterModels(models, { category: 'vision' }).map((m) => m.id)).toEqual(['vision-gguf']);
    expect(filterModels(models, { tag: '转写' }).map((m) => m.id)).toEqual(['audio-ggml']);
    expect(filterModels(models, { tag: 'GGUF' }).map((m) => m.id)).toEqual(['vision-gguf']);
  });

  test('filters by load state and query', () => {
    expect(filterModels(models, { load: 'loaded' }).map((m) => m.id)).toEqual(['chat-onnx']);
    expect(filterModels(models, { load: 'error' }).map((m) => m.id)).toEqual(['audio-ggml']);
    expect(filterModels(models, { query: 'vision' }).map((m) => m.id)).toEqual(['vision-gguf']);
  });

  test('builds stable tag options', () => {
    expect(modelTagOptions(models)).toContain('文本');
    expect(modelTagOptions(models)).toContain('图片');
  });
});
