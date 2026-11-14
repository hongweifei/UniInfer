#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

port="${M1_PORT:-18081}"
if command -v python3 >/dev/null 2>&1; then
  py=python3
else
  py=python
fi
cargo build --quiet -p uniinfer-cli
./target/debug/uniinfer serve \
  --manifest models/tiny-onnx/manifest.toml \
  --addr "127.0.0.1:${port}" >"target/m1-server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null || {
  echo "server did not start" >&2
  cat target/m1-server.log >&2
  exit 1
}

curl -fsS "http://127.0.0.1:${port}/v1/models" \
  | "$py" -c 'import json,sys; data=json.load(sys.stdin); assert data["data"][0]["id"] == "tiny-byte-onnx"' 

body="$(curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-byte-onnx","messages":[{"role":"user","content":"ab"}],"max_tokens":3}')"
echo "$body" \
  | "$py" -c 'import json,sys; body=json.load(sys.stdin); assert body["choices"][0]["message"]["content"] == "cde"' 

stream="$(curl -fsSN "http://127.0.0.1:${port}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"tiny-byte-onnx","messages":[{"role":"user","content":"ab"}],"max_tokens":3,"stream":true}')"
echo "$stream" | grep -q '"content":"c"'
echo "$stream" | grep -q '"content":"d"'
echo "$stream" | grep -q '"content":"e"'
echo "$stream" | grep -q '\[DONE\]'

status="$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:${port}/v1/sessions/unknown/cancel")"
[[ "$status" == "404" ]]

bad="$(mktemp)"
status="$(curl -sS -o "$bad" -w '%{http_code}' \
  "http://127.0.0.1:${port}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d '{"model":')"
[[ "$status" == "400" ]]
"$py" -c 'import json,sys; body=json.load(open(sys.argv[1])); assert body["code"] == "invalid_json"' "$bad"
rm -f "$bad"

metrics="$(curl -fsS "http://127.0.0.1:${port}/metrics")"
echo "$metrics" \
  | "$py" -c 'import json,sys; metrics=json.load(sys.stdin); assert metrics["active_requests"] == 0' 

echo "m1 smoke: ok"
