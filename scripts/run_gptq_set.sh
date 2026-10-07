#!/usr/bin/env bash
# Run GPTQ configs, export into a base GGUF, evaluate.
# Usage: [BASE=models/q/t8-kq-q4k.gguf] run_gptq_set.sh tag:extra,args ...
cd "$(dirname "$0")/.."
BASE=${BASE:-models/q/t8-kq-q4k.gguf}
for spec in "$@"; do
  tag=${spec%%:*}; extra=${spec#*:}; extra=${extra//,/ }
  .venv/bin/python scripts/gptq_ternary.py --tag $tag $extra > results/$tag.log 2>&1
  if [[ ! -f models/gptq/$tag/layer_39.pt ]]; then echo "GPTQ FAILED $tag"; tail -3 results/$tag.log; continue; fi
  .venv/bin/python -I scripts/export_gguf.py $BASE models/gptq/$tag models/q/$tag.gguf > /dev/null &&
    NOBENCH=1 scripts/eval.sh $tag
  rm -f models/q/$tag.gguf models/gptq/$tag/inputs.pt
done
echo SET-DONE
