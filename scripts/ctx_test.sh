#!/usr/bin/env bash
# Start a server config, then measure decode at shallow depth and after a ~240k-token prompt.
# Usage: ctx_test.sh label model.gguf [server args...]
cd "$(dirname "$0")/.."
label=$1 model=$2; shift 2
llama.cpp/build/bin/llama-server -m models/q/$model -ngl 99 -fa on -np 1 --jinja --port 18099 "$@" > results/ctx-$label.log 2>&1 & p=$!
ok=; for i in $(seq 1 90); do curl -s localhost:18099/health | grep -q '"ok"' && { ok=1; break; }; kill -0 $p 2>/dev/null || break; sleep 2; done
[[ -z $ok ]] && { echo "$label: did not start"; grep -E " E " results/ctx-$label.log | head -2; exit 1; }
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
req() { .venv/bin/python - "$1" <<'PY'
import json, sys, time, urllib.request
n = int(sys.argv[1])
code = open("data/calib_code.txt", encoding="utf-8").read()
body = {"temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": False}}
if n == 0:
    body["messages"] = [{"role": "user", "content": "Write a Python LRU cache class. Code only."}]
else:
    body["messages"] = [{"role": "user", "content": code[: int(n * 3.2)] + "\n\nSummarize what the last file above does in one paragraph."}]
req = urllib.request.Request("http://127.0.0.1:18099/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t = json.loads(urllib.request.urlopen(req, timeout=1800).read())
tm = t["timings"]
print(f"prompt {tm['prompt_n']} tok @ {tm['prompt_per_second']:.0f} tok/s, decode {tm['predicted_per_second']:.0f} tok/s"
      + (f", draft acc {tm['draft_n_accepted']}/{tm['draft_n']}" if tm.get("draft_n") else ""))
PY
}
echo "$label | VRAM $vram | shallow: $(req 0) | deep: $(req 240000)"
kill $p; wait $p 2>/dev/null; sleep 2
