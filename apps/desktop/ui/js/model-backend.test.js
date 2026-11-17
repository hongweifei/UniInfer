import { backendChoices, preferredVariant, variantOptions } from './model-backend.js';

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
assert(preferredVariant([{ class: 'cpu', backend: 'cpu' }]) === 'CPU', 'CPU device');
assert(variantOptions(choices[1], 'CPU').join(',') === 'CPU,Vulkan,CUDA', 'configured variant remains visible');
console.log('model-backend tests passed');
