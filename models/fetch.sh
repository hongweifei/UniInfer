#!/usr/bin/env sh
set -eu

dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/tiny-onnx"
cd "$dir"

if [ ! -f model.onnx ] || [ ! -f manifest.toml ]; then
  uv run --quiet --with onnx python generate.py
fi

sha256sum -c sha256.txt
