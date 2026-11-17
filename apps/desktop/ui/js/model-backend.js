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
  if (backend.includes('rocm') || backend.includes('amd')) return 'ROCm';
  return 'Vulkan';
}

export function variantSupport(variant, devices = []) {
  const value = String(variant ?? '').toLowerCase();
  const gpus = devices.filter((device) => device.class === 'gpu');
  const backends = gpus.map((device) => String(device.backend ?? '').toLowerCase());
  if (value === 'cpu') return { supported: true, reason: 'CPU 始终可用' };
  if (value === 'cuda') {
    return backends.some((backend) => backend.includes('cuda') || backend.includes('nvidia'))
      ? { supported: true, reason: '检测到 NVIDIA CUDA 设备' }
      : { supported: false, reason: '需要 NVIDIA CUDA 设备' };
  }
  if (value === 'rocm') {
    return backends.some((backend) => backend.includes('rocm') || backend.includes('amd'))
      ? { supported: true, reason: '检测到 AMD ROCm 设备' }
      : { supported: false, reason: '需要 AMD ROCm 设备' };
  }
  if (value === 'vulkan') {
    return gpus.length
      ? { supported: true, reason: '检测到 GPU，可使用 Vulkan' }
      : { supported: false, reason: '需要 GPU' };
  }
  return { supported: false, reason: '当前设备无法识别这个变体' };
}

export function variantOptions(choice, configured) {
  if (!choice || choice.id === BUILTIN.id) return [];
  const variants = choice.variants ?? [];
  if (configured && configured !== 'auto' && !variants.includes(configured)) {
    return [configured, ...variants];
  }
  return variants;
}
