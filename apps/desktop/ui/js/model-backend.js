const BUILTIN = { id: 'builtin', label: '内置引擎', variants: [] };

const normalize = (value) => String(value ?? '').toLowerCase();
const tokens = (value) => normalize(value).split(/[^a-z0-9]+/).filter(Boolean);

function backendMatches(variant, backend) {
  const variantTokens = tokens(variant);
  const backendTokens = tokens(backend);
  return variantTokens.some((variantToken) =>
    backendTokens.some((backendToken) =>
      backendToken.includes(variantToken) || variantToken.includes(backendToken)
    )
  );
}

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

export function preferredVariant(devices = [], variants = []) {
  const list = variants.filter(Boolean);
  const supported = list.filter((variant) => variantSupport(variant, devices).supported);
  const acceleratorBackends = devices
    .filter((device) => device.class !== 'cpu')
    .map((device) => String(device.backend ?? ''))
    .filter(Boolean);
  for (const backend of acceleratorBackends) {
    const match = supported.find((variant) =>
      !tokens(variant).includes('cpu') && backendMatches(variant, backend)
    );
    if (match) return match;
  }
  return supported.find((variant) => tokens(variant).includes('cpu')) ?? supported[0] ?? 'auto';
}

export function variantSupport(variant, devices = []) {
  const variantTokens = tokens(variant);
  if (!variantTokens.length) return { supported: false, reason: '变体名称未知' };
  const cpu = devices.find((device) => device.class === 'cpu');
  if (variantTokens.includes('cpu')) {
    return cpu
      ? { supported: true, reason: '检测到 CPU' }
      : { supported: false, reason: '需要 CPU' };
  }
  const accelerators = devices.filter((device) => device.class !== 'cpu');
  if (!accelerators.length) return { supported: false, reason: '需要加速设备' };
  const matches = accelerators.filter((device) => backendMatches(variant, device.backend));
  if (matches.length) {
    return {
      supported: true,
      reason: `检测到 ${matches.map((device) => device.backend).filter(Boolean).join(' / ')} 设备`,
    };
  }
  return { supported: false, reason: `未检测到与 ${variant} 匹配的设备后端` };
}

export function variantOptions(choice, configured) {
  if (!choice || choice.id === BUILTIN.id) return [];
  const variants = choice.variants ?? [];
  if (configured && configured !== 'auto' && !variants.includes(configured)) {
    return [configured, ...variants];
  }
  return variants;
}
