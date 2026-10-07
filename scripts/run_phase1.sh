#!/usr/bin/env bash
# Quantize then evaluate each variant in order; Q4_K_M last (needs CPU offload of experts).
cd "$(dirname "$0")/.."
for v in t1-fit t0-absmax iq2xxs q2k t4-fit-q6k t2-fit-q8 t3-fit-q4k t5-fit-q6k-noimx; do
  scripts/quantize_all.sh $v && [[ -f models/q/$v.gguf ]] && scripts/eval.sh $v || echo "EVAL FAILED $v"
done
echo PHASE1-DONE
