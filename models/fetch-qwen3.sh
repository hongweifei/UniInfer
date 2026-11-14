#!/usr/bin/env bash
# Fetch Qwen3-0.6B Q4_K_M GGUF and tokenizer from ModelScope.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/qwen3-0.6b"
mkdir -p "$dest"

gguf=Qwen3-0.6B-Q4_K_M.gguf
base_gguf=https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master
base_tok=https://modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master

curl -fSL --retry 3 "$base_gguf/$gguf" -o "$dest/model.gguf"
curl -fSL --retry 3 "$base_tok/tokenizer.json" -o "$dest/tokenizer.json"
curl -fSL --retry 3 "$base_tok/tokenizer_config.json" -o "$dest/tokenizer_config.json"

cat <<SUMS | sha256sum -c -
ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  $dest/model.gguf
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  $dest/tokenizer.json
d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101  $dest/tokenizer_config.json
SUMS
echo "qwen3 ready: $dest"
