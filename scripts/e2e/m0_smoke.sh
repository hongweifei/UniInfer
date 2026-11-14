#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

cargo run --quiet -p uniinfer-cli -- devices

output="$(cargo run --quiet -p uniinfer-cli -- run \
  --manifest models/tiny-onnx/manifest.toml \
  --prompt ab \
  --max-tokens 3)"

if [[ "$output" != "cde" ]]; then
  echo "unexpected generation: '$output'" >&2
  exit 1
fi

cargo test --quiet -p uniinfer-engine-onnx --test engine_contract
echo "m0 smoke: ok"
