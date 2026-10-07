#!/usr/bin/env bash
# Coding evals against a llama-server per model. Usage: eval_coding.sh humaneval|suite model [model...]
# Q4_K_M runs with experts of 16 layers on the CPU (does not fit in 16 GB).
cd "$(dirname "$0")/.."
B=${LLAMA_BIN:-llama.cpp/build/bin}; BONSAI=~/dev/bonsai-ada-surgery; PY=$PWD/.venv/bin/python; PORT=18090
what=$1; shift
for m in "$@"; do
  extra=(); [[ $m == bonsai-27b ]] && extra=(--chat-template-file $HOME/.local/opt/llama-prism/bonsai-chat-template.jinja)
  [[ $m == q4km ]] && extra=(--n-cpu-moe 16 --no-op-offload)  # op offload of CPU experts crashes on long prompts
  $B/llama-server -m models/q/$m.gguf -ngl 99 -fa on -c 65536 -ctk q8_0 -ctv q8_0 -np 1 --jinja \
    --port $PORT --alias $m "${extra[@]}" > results/server-$m.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do curl -s localhost:$PORT/health | grep -q '"ok"' && break; sleep 2; done
  out=results/$what/$m; mkdir -p $out
  if [[ $what == humaneval ]]; then
    (cd $BONSAI && $PY bench/humaneval_run.py --base http://127.0.0.1:$PORT --key-file "" --arm off \
      --problems ~/dev/ternary-qwen36/data/human-eval/HumanEval.jsonl.gz --out ~/dev/ternary-qwen36/$out) > $out/run.log 2>&1
    tail -3 $out/run.log
  else
    (cd $BONSAI && $PY suite/run_suite.py --base http://127.0.0.1:$PORT --key-file "" --model $m ${SUITE_ARGS:-} --out ~/dev/ternary-qwen36/$out) > $out/run.log 2>&1
    tail -5 $out/run.log
  fi
  kill $pid; wait $pid 2>/dev/null
done
