#!/usr/bin/env bash
# Perplexity (wikitext-2, code) and speed for each quantized variant. Appends to results/eval.tsv.
# Usage: [NCMOE=n] [NOBENCH=1] scripts/eval.sh variant [ngl]
set -euo pipefail
cd "$(dirname "$0")/.."
B=llama.cpp/build/bin
v=$1; ngl=${2:-99}; f=models/q/$v.gguf
mkdir -p results/eval-logs
ppl() { # ppl corpus chunks
  $B/llama-perplexity -m $f -f $1 -c 2048 --chunks $2 -ngl $ngl -fa on -b 2048 ${NCMOE:+--n-cpu-moe $NCMOE} 2>&1 | tee results/eval-logs/$v-$(basename $1).log | grep -oP 'Final estimate: PPL = \K[0-9.]+ \+/- [0-9.]+'
}
wiki=$(ppl data/wikitext-2-raw/wiki.test.raw 40)
code=$(ppl data/code.txt 40)
speed=-; [[ -n ${NOBENCH:-} ]] || speed=$($B/llama-bench -m $f -ngl $ngl ${NCMOE:+-ncmoe $NCMOE} -fa 1 -p 512 -n 128 -r 3 -o csv 2>/dev/null | python3 -c "
import csv,sys; r=list(csv.DictReader(sys.stdin)); print(' '.join(f\"{x['n_prompt']}/{x['n_gen']}={float(x['avg_ts']):.1f}\" for x in r))")
size=$(du -m $f | cut -f1)
[[ -f results/eval.tsv ]] || printf 'variant\tsize_mb\tppl_wiki\tppl_code\tspeed_tps\n' > results/eval.tsv
printf '%s\t%s\t%s\t%s\t%s\n' "$v" "$size" "$wiki" "$code" "$speed" | tee -a results/eval.tsv
