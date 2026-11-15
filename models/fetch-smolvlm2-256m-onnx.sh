#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/models/smolvlm2-256m-onnx-q4"
base="https://hf-mirror.com/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/resolve/main"
mkdir -p "$dest"
for file in vision_encoder_q4.onnx embed_tokens_q4.onnx decoder_model_merged_q4.onnx; do
  curl -k -L --retry 3 -o "$dest/$file" "$base/onnx/$file"
done
for file in tokenizer.json tokenizer_config.json config.json generation_config.json preprocessor_config.json; do
  curl -k -L --retry 3 -o "$dest/$file" "$base/$file"
done
python - "$dest" <<'PY'
import json, sys
from pathlib import Path
dest = Path(sys.argv[1])
value = json.loads((dest / "preprocessor_config.json").read_text())
value["do_image_splitting"] = False
value["size"] = {"longest_edge": 512}
value["max_image_size"] = {"longest_edge": 512}
(dest / "preprocessor_config_512.json").write_text(json.dumps(value, indent=2) + "\n")
PY
sha256sum "$dest"/*.onnx
