import { backendChoices, preferredVariant, variantOptions, variantSupport } from './model-backend.js';

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const backends = [
  { kind: 'builtin', name: 'llama.cpp', engines: [{ variant: '内置' }] },
  { kind: 'sidecar', name: 'llama.cpp', engines: [{ variant: 'Vulkan' }] },
  { kind: 'sidecar', name: 'llama.cpp', engines: [{ variant: 'CUDA' }] },
  { kind: 'sidecar', name: 'stable-diffusion.cpp', engines: [{ variant: 'CPU' }] },
];
const choices = backendChoices(backends);
assert(choices[0].id === 'builtin', 'builtin must be first');
assert(choices.length === 3, 'sidecars group by family');
assert(choices[1].variants.join(',') === 'Vulkan,CUDA', 'variants are unique');
assert(preferredVariant([{ class: 'gpu', backend: 'cuda' }], ['Vulkan', 'CUDA 12.4', 'CPU']) === 'CUDA 12.4', 'CUDA device');
assert(preferredVariant([{ class: 'gpu', backend: 'vulkan' }], ['Vulkan', 'CPU']) === 'Vulkan', 'Vulkan device');
assert(preferredVariant([{ class: 'gpu', backend: 'rocm' }], ['ROCm 10.0', 'CPU']) === 'ROCm 10.0', 'ROCm device');
assert(preferredVariant([{ class: 'cpu', backend: 'cpu' }], ['Vulkan', 'CPU']) === 'CPU', 'CPU device');
assert(variantOptions(choices[1], 'CPU').join(',') === 'CPU,Vulkan,CUDA', 'configured variant remains visible');
assert(variantSupport('CPU', [{ class: 'cpu', backend: 'cpu' }]).supported, 'CPU supported');
assert(variantSupport('CUDA 12.4', [{ class: 'gpu', backend: 'directml' }], [{ vendor: 'NVIDIA', model: 'RTX' }]).supported, 'CUDA via NVIDIA vendor');
assert(!variantSupport('CUDA 12.4', [{ class: 'gpu', backend: 'directml' }], [{ vendor: 'AMD', model: 'Radeon' }]).supported, 'CUDA rejected on AMD');
assert(variantSupport('ROCm 10.0', [{ class: 'gpu', backend: 'directml' }], [{ vendor: 'AMD', model: 'Radeon' }]).supported, 'ROCm via AMD vendor');
assert(variantSupport('SYCL', [{ class: 'gpu', backend: 'directml' }], [{ vendor: 'Intel', model: 'Arc' }]).supported, 'new variants are accepted generically');
assert(variantSupport('CUDA 12.4', [{ class: 'gpu', backend: 'cuda' }]).supported, 'CUDA device');
assert(!variantSupport('CUDA 12.4', [{ class: 'gpu', backend: 'vulkan' }], [{ vendor: 'AMD' }]).supported, 'CUDA rejected on AMD');
assert(variantSupport('ROCm 10.0', [{ class: 'gpu', backend: 'rocm' }]).supported, 'ROCm device');
assert(variantSupport('SYCL', [{ class: 'gpu', backend: 'sycl' }]).supported, 'SYCL is discovered generically');
assert(variantSupport('SYCL', [{ class: 'gpu', backend: 'vulkan' }]).supported, 'new variant is allowed with a GPU');
assert(variantSupport('Vulkan', [{ class: 'gpu', backend: 'vulkan' }]).supported, 'Vulkan device');
assert(!variantSupport('Vulkan', []).supported, 'Vulkan needs GPU');
console.log('model-backend tests passed');
