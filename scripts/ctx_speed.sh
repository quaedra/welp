#!/usr/bin/env bash
# Decode and prefill speed against context depth (llama-bench -d), one run per depth so a depth that
# does not fit in 16 GB is recorded as such instead of ending the sweep.
# Same flags for every model: flash attention, q4_0 KV (needed for 262k on 16 GB), -ub 256.
# tg128 = decode speed after <depth> tokens of context; pp2048 = prefill of the next 2048 tokens.
# Usage: scripts/ctx_speed.sh [model...]   (waits until the GPU is free)
cd "$(dirname "$0")/.."
B=${LLAMA_BIN:-llama.cpp/build/bin}; out=results/ctxspeed; mkdir -p $out
models=("${@:-ig-gptq qwen38-27b-q3kxl bonsai-27b}"); models=(${models[@]})
depths=(0 4096 16384 32768 65536 131072 196608 258048)

until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) < 2000 )) && ! pgrep -x llama-server > /dev/null; do
  sleep 60
done
echo "$(date +%H:%M) GPU free, starting"

for m in "${models[@]}"; do
  for d in "${depths[@]}"; do
    f=$out/$m-d$d.jsonl
    [[ -s $f ]] && continue
    echo "$(date +%H:%M) $m depth $d"
    if ! $B/llama-bench -m models/q/$m.gguf -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -ub 256 \
        -d $d -p 2048 -n 128 -r 2 -o jsonl > $f.tmp 2> $out/$m-d$d.log; then
      echo "$(date +%H:%M) $m depth $d failed (does not fit?)"; rm -f $f.tmp; echo '{"failed": true}' > $out/$m-d$d.fail
      break  # deeper depths will not fit either
    fi
    mv $f.tmp $f
  done
done
echo CTX-SPEED-DONE
