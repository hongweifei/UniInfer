import { modelCategories, modelTags } from './model-meta.js';

export function filterModels(models, { category = 'all', tag = 'all', format = 'all', load = 'all', query = '' } = {}) {
  const needle = String(query ?? '').trim().toLowerCase();
  return models.filter((model) => {
    if (category !== 'all' && !modelCategories(model).includes(category)) return false;
    if (tag !== 'all' && !modelTags(model).includes(tag)) return false;
    if (format !== 'all' && model.format !== format) return false;
    if (load === 'loaded' && !model.loaded) return false;
    if (load === 'idle' && model.loaded) return false;
    if (load === 'error' && !model.route_error) return false;
    if (!needle) return true;
    const text = [
      model.id, model.format, model.quantization, model.runtime, model.implementation,
      ...(model.inputs ?? []), ...(model.outputs ?? []), ...modelTags(model),
    ].join(' ').toLowerCase();
    return text.includes(needle);
  });
}

export function modelTagOptions(models) {
  return [...new Set(models.flatMap((model) => modelTags(model)))]
    .sort((a, b) => a.localeCompare(b, 'zh-CN'));
}
