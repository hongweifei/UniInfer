#!/usr/bin/env bash
# Fetch the int4 ONNX Qwen3-0.6B GenAI bundle from ModelScope.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/qwen3-0.6b-onnx"
base="https://modelscope.cn/api/v1/models/onnx-community/Qwen3-0.6B-ONNX/repo?Revision=master&FilePath=onnxruntime%2Fcpu_and_mobile%2Fcpu-int4-kld-block-128%2F"

mkdir -p "$dest"
for file in model.onnx genai_config.json tokenizer.json tokenizer_config.json config.json; do
  if [[ -f "$dest/$file" ]]; then
    echo "skip $file"
  else
    echo "fetch $file"
    curl -fSL --retry 3 "$base$file" -o "$dest/$file"
  fi
done
sha256sum "$dest/model.onnx"
