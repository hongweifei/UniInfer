#!/usr/bin/env bash
# Fetch Whisper tiny.en ONNX encoder/decoder from ModelScope.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/whisper-tiny.en-onnx"
base="https://modelscope.cn/models/onnx-community/whisper-tiny.en-ONNX/resolve/master"
mkdir -p "$dest/onnx"

for file in onnx/encoder_model.onnx onnx/decoder_model.onnx tokenizer.json config.json; do
  if [[ ! -f "$dest/$file" ]]; then
    mkdir -p "$(dirname "$dest/$file")"
    curl -fSL --retry 3 "$base/$file" -o "$dest/$file"
  fi
done

cat <<SUMS | sha256sum -c -
ee247387567b1f9dea4d6f7d077d852ff2648038e5f50ecb6a744aff3db74856  $dest/onnx/encoder_model.onnx
018fba571ba7408856ae35c1d5577b6eeb4905d2fdec9bd20b9f8debb815bf96  $dest/onnx/decoder_model.onnx
1bb34b47daba40728f488323190dc64765c5338b5196f95c4af926e13a6161c3  $dest/tokenizer.json
79e11e4a0d1415f68dcfaa32e8dbeda0856d44dd1954f0a075867885fa759a70  $dest/config.json
SUMS
echo "whisper onnx ready: $dest"
