#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/qwen3-vl-2b"
mkdir -p "$dest"
uvx --from modelscope-hub modelscope download \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --local_dir "$dest"
