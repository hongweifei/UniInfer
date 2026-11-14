#!/usr/bin/env bash
# Fetch Qwen3-0.6B Safetensors (BF16) from ModelScope.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/qwen3-0.6b-safetensors"
mkdir -p "$dest"
base=https://modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master

for file in model.safetensors config.json tokenizer.json tokenizer_config.json; do
  curl -fSL --retry 3 "$base/$file" -o "$dest/$file"
done

cat <<SUMS | sha256sum -c -
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  $dest/model.safetensors
SUMS
echo "qwen3 safetensors ready: $dest"
