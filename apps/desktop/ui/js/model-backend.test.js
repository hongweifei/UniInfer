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
assert(preferredVariant([{ class: 'gpu', backend: 'cuda' }]) === 'CUDA', 'CUDA device');
assert(preferredVariant([{ class: 'gpu', backend: 'vulkan' }]) === 'Vulkan', 'Vulkan device');
assert(preferredVariant([{ class: 'gpu', backend: 'rocm' }]) === 'ROCm', 'ROCm device');
assert(preferredVariant([{ class: 'cpu', backend: 'cpu' }]) === 'CPU', 'CPU device');
assert(variantOptions(choices[1], 'CPU').join(',') === 'CPU,Vulkan,CUDA', 'configured variant remains visible');
assert(variantSupport('CPU', []).supported, 'CPU always supported');
assert(variantSupport('CUDA', [{ class: 'gpu', backend: 'cuda' }]).supported, 'CUDA device');
assert(!variantSupport('CUDA', [{ class: 'gpu', backend: 'vulkan' }]).supported, 'CUDA needs NVIDIA');
assert(variantSupport('ROCm', [{ class: 'gpu', backend: 'rocm' }]).supported, 'ROCm device');
assert(!variantSupport('ROCm', []).supported, 'ROCm needs AMD');
assert(variantSupport('Vulkan', [{ class: 'gpu', backend: 'cuda' }]).supported, 'Vulkan can use NVIDIA GPU');
assert(!variantSupport('Vulkan', []).supported, 'Vulkan needs GPU');
console.log('model-backend tests passed');
