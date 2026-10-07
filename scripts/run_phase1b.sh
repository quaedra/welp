#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for v in t6-fit-q6k-imxsqrt t7-fit-q6k-imxkq; do
  scripts/quantize_all.sh $v && NOBENCH=1 scripts/eval.sh $v || echo "EVAL FAILED $v"
done
scripts/quantize_all.sh q4km && NCMOE=16 scripts/eval.sh q4km || echo "EVAL FAILED q4km"
echo PHASE1B-DONE
