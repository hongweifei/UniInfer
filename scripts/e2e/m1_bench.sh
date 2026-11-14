#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mkdir -p target

if command -v python3 >/dev/null 2>&1; then
  py=python3
else
  py=python
fi

cargo run --quiet -p uniinfer-bench -- run \
  --manifest models/tiny-onnx/manifest.toml \
  --prompt ab \
  --max-tokens 3 \
  --iterations 3 \
  --save target/bench-baseline.json > target/bench-current.json

"$py" - <<'PY'
import json
with open("target/bench-current.json", encoding="utf-8") as handle:
    data = json.load(handle)
assert data["schema"] == 1
assert data["model"] == "tiny-byte-onnx"
assert data["generated_bytes"] == 9
assert data["ttft_ms"] > 0
assert data["total_ms"] >= data["ttft_ms"]
assert data["tokens_per_second"] > 0
assert data["peak_rss_bytes"] > 0
PY

cargo run --quiet -p uniinfer-bench -- compare \
  --baseline target/bench-baseline.json \
  --current target/bench-current.json

echo "m1 bench: ok"
