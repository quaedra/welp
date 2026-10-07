#!/usr/bin/env bash
# Download Unsloth UD-IQ3_XXS, wait for the suite run to finish, then HumanEval + perplexity + max context in 16 GB.
cd "$(dirname "$0")/.."
curl -sSL --fail --retry 20 -C - -o models/q/ud-iq3xxs.gguf \
  https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf || exit 1
echo downloaded
while pgrep -f "eval_coding.sh suite" > /dev/null; do sleep 60; done
scripts/eval_coding.sh humaneval ud-iq3xxs
scripts/eval.sh ud-iq3xxs
for ctx in 262144 131072 65536; do
  llama.cpp/build/bin/llama-server -m models/q/ud-iq3xxs.gguf -ngl 99 -fa on -c $ctx -ctk q8_0 -ctv q8_0 --port 18099 > results/vram-iq3-$ctx.log 2>&1 & p=$!
  ok=; for i in $(seq 1 60); do curl -s localhost:18099/health | grep -q '"ok"' && { ok=1; break; }; kill -0 $p 2>/dev/null || break; sleep 2; done
  echo "ctx $ctx: ${ok:+fits} $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  kill $p 2>/dev/null; wait $p 2>/dev/null; sleep 2
  [[ -n $ok ]] && break
done
echo IQ3-DONE
