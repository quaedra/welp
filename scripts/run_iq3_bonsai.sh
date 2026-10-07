#!/usr/bin/env bash
# IQ3_XXS: HumanEval, perplexity, largest context that fits fully on the GPU. Then Bonsai 27B HumanEval.
cd "$(dirname "$0")/.."
scripts/eval_coding.sh humaneval ud-iq3xxs
scripts/eval.sh ud-iq3xxs
for ctx in 262144 131072 65536; do
  llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs.gguf -ngl 99 -fa on -c $ctx -ctk q8_0 -ctv q8_0 --port 18099 > results/vram-iq3-$ctx.log 2>&1 & p=$!
  ok=; for i in $(seq 1 60); do curl -s localhost:18099/health | grep -q '"ok"' && { ok=1; break; }; kill -0 $p 2>/dev/null || break; sleep 2; done
  echo "ctx $ctx: ${ok:+up} $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $p 2>/dev/null; wait $p 2>/dev/null; sleep 2
done
scripts/eval_coding.sh humaneval bonsai-27b
echo ALL-DONE
