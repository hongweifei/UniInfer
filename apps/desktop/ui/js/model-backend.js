const BUILTIN = { id: 'builtin', label: '内置引擎', variants: [] };

export function backendChoices(backends = []) {
  const groups = new Map();
  for (const backend of backends) {
    if (backend.kind === 'builtin') continue;
    const id = String(backend.name ?? '').trim();
    if (!id) continue;
    let group = groups.get(id);
    if (!group) {
      group = { id, label: id, variants: [] };
      groups.set(id, group);
    }
    for (const engine of backend.engines ?? []) {
      const variant = String(engine.variant ?? '').trim();
      if (variant && !group.variants.includes(variant)) group.variants.push(variant);
    }
  }
  return [BUILTIN, ...groups.values()];
}

export function preferredVariant(devices = []) {
  const gpu = devices.find((device) => device.class === 'gpu');
  if (!gpu) return 'CPU';
  const backend = String(gpu.backend ?? '').toLowerCase();
  if (backend.includes('cuda') || backend.includes('nvidia')) return 'CUDA';
  return 'Vulkan';
}

export function variantOptions(choice, configured) {
  if (!choice || choice.id === BUILTIN.id) return [];
  const variants = choice.variants ?? [];
  if (configured && configured !== 'auto' && !variants.includes(configured)) {
    return [configured, ...variants];
  }
  return variants;
}
