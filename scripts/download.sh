#!/usr/bin/env bash
# Resumable downloads of the BF16 source GGUF and Unsloth's imatrix.
set -euo pipefail
cd "$(dirname "$0")/../models"
BASE=https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main
for f in imatrix_unsloth.gguf_file \
         BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
         BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf; do
  out=$(basename "$f")
  echo "== $out"
  curl -L --fail --retry 20 --retry-delay 5 -C - -o "$out" "$BASE/$f"
done
echo DONE
