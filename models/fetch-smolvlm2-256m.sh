#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/smolvlm2-256m-q4"
base="https://hf-mirror.com/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main"
hf="https://hf-mirror.com/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/resolve/main"
mkdir -p "$dest"
for file in \
  SmolVLM2-256M-Video-Instruct-Q4_K_M.gguf \
  mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf
do
  curl -k -L --retry 3 -o "$dest/$file" "$base/$file"
done
for file in tokenizer.json tokenizer_config.json preprocessor_config.json; do
  curl -k -L --retry 3 -o "$dest/$file" "$hf/$file"
done
sha256sum "$dest"/*.gguf "$dest"/tokenizer.json "$dest"/tokenizer_config.json
