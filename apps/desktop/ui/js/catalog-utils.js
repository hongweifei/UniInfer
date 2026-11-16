import { onlineCategories, onlineTags } from './model-meta.js';

export function filterOnlineItems(items, { category = 'all', tag = 'all' } = {}) {
  return items.filter((entry) => {
    if (category !== 'all' && !onlineCategories(entry).includes(category)) return false;
    if (tag !== 'all' && !onlineTags(entry).includes(tag)) return false;
    return true;
  });
}

export function onlineFacets(items) {
  const categories = new Map();
  const tags = new Set();
  for (const entry of items) {
    for (const category of onlineCategories(entry)) {
      categories.set(category, (categories.get(category) ?? 0) + 1);
    }
    for (const tag of onlineTags(entry)) tags.add(tag);
  }
  return {
    categories,
    tags: [...tags].sort((a, b) => a.localeCompare(b, 'zh-CN')),
  };
}

export function clampPage(page, totalResults, pageSize) {
  const target = Number.parseInt(page, 10);
  if (!Number.isFinite(target) || target < 1) return null;
  if (!totalResults || !pageSize) return target;
  return Math.min(target, Math.max(1, Math.ceil(totalResults / pageSize)));
}
