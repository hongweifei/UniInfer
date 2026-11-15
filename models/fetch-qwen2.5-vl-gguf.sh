#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/qwen2.5-vl-7b-gguf"
mkdir -p "$dest"
uvx --from modelscope-hub modelscope download \
  --model ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
  --local_dir "$dest" \
  --include "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
