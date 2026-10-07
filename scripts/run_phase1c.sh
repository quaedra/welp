#!/usr/bin/env bash
cd "$(dirname "$0")/.."
scripts/quantize_all.sh t8-kq-q4k && scripts/eval.sh t8-kq-q4k || echo "EVAL FAILED t8"
for v in g4 g0 g1 g2 g3 tdown; do
  scripts/quantize_all.sh $v && NOBENCH=1 scripts/eval.sh $v || echo "EVAL FAILED $v"
  rm -f models/q/$v.gguf   # keep disk free; numbers are in results/eval.tsv
done
echo PHASE1C-DONE
