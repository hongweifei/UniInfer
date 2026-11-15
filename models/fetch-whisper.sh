#!/usr/bin/env bash
# Fetch Whisper large-v3-turbo GGML from ModelScope.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/whisper-large-v3-turbo-ggml"
base="https://modelscope.cn/models/ivrit-ai/yi-whisper-large-v3-turbo-ggml/resolve/master"
mkdir -p "$dest"

if [[ ! -f "$dest/ggml-model.bin" ]]; then
  curl -fSL --retry 3 "$base/ggml-model.bin" -o "$dest/ggml-model.bin"
fi
cat <<SUMS | sha256sum -c -
a2094962a33f48ce1fc59ab6f34ca9bb4408443bf8a86361b6b1775aaa4bc1f3  $dest/ggml-model.bin
SUMS
echo "whisper ready: $dest"
