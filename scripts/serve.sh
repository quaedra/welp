#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B UD-IQ3_XXS on 16 GB.
# MODE=full (default): 262k ctx, q4_0 KV, ~155 tok/s shallow / ~107 at 214k. long: 131k ctx, q8_0 KV. fast: 64k ctx, q8_0 KV, MTP draft (~210 tok/s).
cd "$(dirname "$0")/.."
MODE=${MODE:-full}; PORT=${PORT:-8080}; HOST=${HOST:-127.0.0.1}
common=(-ngl 99 -fa on -np 1 --jinja --host "$HOST" --port "$PORT" --alias qwen3.6-35b-a3b)
case $MODE in
  full) exec llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs.gguf -c 262144 -ctk q4_0 -ctv q4_0 -ub 256 "${common[@]}" ;;
  long) exec llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs.gguf -c 131072 -ctk q8_0 -ctv q8_0 "${common[@]}" ;;
  fast) exec llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs-mtp.gguf -c 65536 -ctk q8_0 -ctv q8_0 "${common[@]}" \
          --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0 ;;
  *) echo "MODE must be full, long or fast" >&2; exit 1 ;;
esac
