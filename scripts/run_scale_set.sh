#!/usr/bin/env bash
# Scale-tune GPTQ trits, export into a base GGUF, evaluate. Usage: run_scale_set.sh SRC TAG BASE [extra args]
cd "$(dirname "$0")/.."
src=$1 tag=$2 base=$3; shift 3
(cd scripts && ../.venv/bin/python scale_tune.py --src $src --tag $tag "$@") > results/$tag.log 2>&1
if [[ ! -f models/gptq/$tag/layer_39.pt ]]; then echo "TUNE FAILED $tag"; tail -3 results/$tag.log; exit 1; fi
.venv/bin/python -I scripts/export_gguf.py $base models/gptq/$tag models/q/$tag.gguf > /dev/null && NOBENCH=1 scripts/eval.sh $tag
