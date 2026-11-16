import { describe, expect, test } from 'bun:test';
import { clampPage, filterOnlineItems, onlineFacets } from './catalog-utils.js';

const items = [
  { id: 'a', task: 'text-generation', tags: ['gguf'] },
  { id: 'b', task: 'text-to-image', tags: ['diffusion', 'safetensors'] },
  { id: 'c', task: 'text-to-video', tags: ['diffusion'] },
];

describe('online catalog filters', () => {
  test('filters by category and tag', () => {
    expect(filterOnlineItems(items, { category: 'image' }).map((entry) => entry.id)).toEqual(['b']);
    expect(filterOnlineItems(items, { category: 'video' }).map((entry) => entry.id)).toEqual(['c']);
    expect(filterOnlineItems(items, { tag: 'gguf' }).map((entry) => entry.id)).toEqual(['a']);
  });

  test('computes facet counts', () => {
    const facets = onlineFacets(items);
    expect(facets.categories.get('image')).toBe(1);
    expect(facets.categories.get('video')).toBe(1);
    expect(facets.tags).toContain('diffusion');
  });

  test('clamps page numbers', () => {
    expect(clampPage('0', 100, 10)).toBeNull();
    expect(clampPage('5', 100, 10)).toBe(5);
    expect(clampPage('99', 100, 10)).toBe(10);
    expect(clampPage('5', null, 10)).toBe(5);
  });
});
