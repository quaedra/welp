#!/usr/bin/env bash
# gptq_ggml runs exported into the UD-IQ3_XXS base, then perplexity. Each arg: tag:extra,args
cd "$(dirname "$0")/.."
for spec in "$@"; do
  tag=${spec%%:*}; extra=${spec#*:}; extra=${extra//,/ }
  (cd scripts && ../.venv/bin/python gptq_ggml.py --tag $tag $extra) > results/$tag.log 2>&1
  if [[ ! -f models/gptq/$tag/layer_39.npz ]]; then echo "FAILED $tag"; tail -3 results/$tag.log; continue; fi
  .venv/bin/python -I scripts/export_ggml.py models/q/ud-iq3xxs.gguf models/gptq/$tag models/q/$tag.gguf > /dev/null && NOBENCH=1 scripts/eval.sh $tag
done
echo SET-DONE
