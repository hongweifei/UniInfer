#!/usr/bin/env bash
# Download a model from ModelScope into models/<name> by default.
# Usage: models/pull.sh <namespace/model> [local_dir] [revision]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <namespace/model> [local_dir] [revision]" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="$1"
dest="${2:-models/$(basename "$model")}"
revision="${3:-}"

mkdir -p "$root/$(dirname "$dest")"
abs="$(cd "$root/$(dirname "$dest")" && pwd)/$(basename "$dest")"

if command -v modelscope >/dev/null 2>&1; then
  downloader=(modelscope)
else
  downloader=(uvx --from modelscope modelscope)
fi

args=(download --model "$model" --local_dir "$abs")
if [[ -n "$revision" ]]; then
  args+=(--revision "$revision")
fi

"${downloader[@]}" "${args[@]}"
echo "downloaded $model -> $abs"
