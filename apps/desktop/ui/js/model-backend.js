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
  const builtin = { ...BUILTIN, variants: [] };
  for (const backend of backends) {
    if (backend.kind === 'builtin') {
      for (const engine of backend.engines ?? []) {
        const variants = Array.isArray(engine.variants) && engine.variants.length
          ? engine.variants
          : (engine.devices ?? []).map((device) => String(device).toUpperCase());
        for (const variant of variants) {
          const label = String(variant).trim();
          if (label && !builtin.variants.includes(label)) builtin.variants.push(label);
        }
      }
      continue;
    }
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
  return [builtin, ...groups.values()];
}

export function preferredVariant(devices = [], variants = [], graphics = []) {
  const list = variants.filter(Boolean);
  const supported = list.filter((variant) => variantSupport(variant, devices, graphics).supported);
  const backends = devices
    .filter((device) => device.class !== 'cpu')
    .map((device) => String(device.backend ?? ''))
    .filter(Boolean);
  const direct = supported.find((variant) =>
    !tokens(variant).includes('cpu') && backends.some((backend) => backendMatches(variant, backend))
  );
  if (direct) return direct;
  const hardware = hardwareText(graphics);
  if (hardware) {
    const vendor = supported.find((variant) =>
      tokens(variant).some((token) =>
        (token === 'cuda' && vendorMatches(hardware, 'nvidia'))
        || (token === 'rocm' && vendorMatches(hardware, 'amd'))
      )
    );
    if (vendor) return vendor;
  }
  const cpu = supported.find((variant) => tokens(variant).includes('cpu'));
  const accelerator = supported.find((variant) => !tokens(variant).includes('cpu'));
  return accelerator ?? cpu ?? supported[0] ?? 'auto';
}

function hardwareText(graphics) {
  return graphics
    .map((gpu) => `${gpu.vendor ?? ''} ${gpu.model ?? ''}`)
    .join(' ')
    .toLowerCase();
}

function vendorMatches(hardware, vendor) {
  if (vendor === 'amd') {
    return hardware.includes('amd') || hardware.includes('advanced micro devices') || hardware.includes('radeon');
  }
  return hardware.includes(vendor);
}

export function variantSupport(variant, devices = [], graphics = []) {
  const variantTokens = tokens(variant);
  if (!variantTokens.length) return { supported: false, reason: '变体名称未知' };
  const cpu = devices.find((device) => device.class === 'cpu');
  if (variantTokens.includes('cpu')) {
    return cpu
      ? { supported: true, reason: '检测到 CPU' }
      : { supported: false, reason: '需要 CPU' };
  }
  const accelerators = devices.filter((device) => device.class !== 'cpu');
  const backends = accelerators.map((device) => String(device.backend ?? '')).filter(Boolean);
  if (backends.some((backend) => backendMatches(variant, backend))) {
    return { supported: true, reason: `检测到 ${backends.join(' / ')} 后端` };
  }
  const hardware = hardwareText(graphics);
  const hasAccelerator = accelerators.length > 0 || graphics.length > 0;
  if (!hasAccelerator) return { supported: false, reason: '需要 GPU/NPU' };

  // Vendor-exclusive APIs. Every other variant is considered downloadable
  // whenever an accelerator is present; no variant allowlist is maintained.
  const vendorExclusive = variantTokens
    .map((token) => token === 'cuda' ? 'nvidia' : token === 'rocm' ? 'amd' : null)
    .find(Boolean);
  if (vendorExclusive && hardware) {
    return vendorMatches(hardware, vendorExclusive)
      ? { supported: true, reason: `检测到 ${vendorExclusive.toUpperCase()} GPU` }
      : { supported: false, reason: `需要 ${vendorExclusive.toUpperCase()} GPU` };
  }
  return { supported: true, reason: '检测到加速设备' };
}

export function variantOptions(choice, configured) {
  if (!choice || choice.id === BUILTIN.id) return [];
  const variants = choice.variants ?? [];
  if (configured && configured !== 'auto' && !variants.includes(configured)) {
    return [configured, ...variants];
  }
  return variants;
}
