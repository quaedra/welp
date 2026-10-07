#!/usr/bin/env bash
# Resumable download of the original Qwen3.6-35B-A3B safetensors (text weights + configs).
set -euo pipefail
mkdir -p "$(dirname "$0")/../models/hf" && cd "$(dirname "$0")/../models/hf"
BASE=https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main
files=(config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja model.safetensors.index.json)
for i in $(seq -w 1 26); do files+=("model-000$i-of-00026.safetensors"); done
for f in "${files[@]}"; do
  echo "== $f"
  curl -sS -L --fail --retry 20 --retry-delay 5 -C - -o "$f" "$BASE/$f"
done
echo DONE
