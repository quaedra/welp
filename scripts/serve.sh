#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B UD-IQ3_XXS on 16 GB. MODE=long (131k ctx, no MTP, ~160 tok/s) or fast (64k ctx, MTP draft, ~210 tok/s).
cd "$(dirname "$0")/.."
MODE=${MODE:-long}; PORT=${PORT:-8080}; HOST=${HOST:-127.0.0.1}
common=(-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja --host "$HOST" --port "$PORT" --alias qwen3.6-35b-a3b)
case $MODE in
  long) exec llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs.gguf -c 131072 "${common[@]}" ;;
  fast) exec llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs-mtp.gguf -c 65536 "${common[@]}" \
          --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0 ;;
  *) echo "MODE must be long or fast" >&2; exit 1 ;;
esac
