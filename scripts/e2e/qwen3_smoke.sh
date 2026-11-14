#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

if [[ ! -f models/qwen3-0.6b/model.gguf ]]; then
  echo "missing model; run models/fetch-qwen3.sh" >&2
  exit 2
fi

output="$(cargo run --quiet -p uniinfer-cli -- run \
  --manifest models/qwen3-0.6b/manifest.toml \
  --prompt "Hello, who are you?" \
  --max-tokens 16)"

if [[ -z "$output" ]]; then
  echo "empty generation" >&2
  exit 1
fi

echo "qwen3 smoke: ok"
