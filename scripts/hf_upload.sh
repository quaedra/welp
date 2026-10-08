#!/usr/bin/env bash
# Upload release/hf to Hugging Face over a flaky connection: retry until the GGUF lands, pausing between
# attempts and waiting for the network to come back, then check the uploaded file's SHA-256.
# Usage: scripts/hf_upload.sh [max_attempts]   (run detached: setsid nohup scripts/hf_upload.sh > results/hf-upload.log 2>&1 &)
cd "$(dirname "$0")/.."
REPO=quaedra/Welp-35B-A3B-GGUF; FILE=Welp-35B-A3B.gguf; MAX=${1:-50}
HF=$PWD/.venv/bin/hf
# Keep retrying a failed Xet request instead of failing the whole upload; fewer parallel streams for Wi-Fi.
export HF_XET_CLIENT_RETRY_MAX_ATTEMPTS=${HF_XET_CLIENT_RETRY_MAX_ATTEMPTS:-1000}
export HF_XET_FIXED_UPLOAD_CONCURRENCY=${HF_XET_FIXED_UPLOAD_CONCURRENCY:-4}

want=$(awk -v f=$FILE '$2 == f {print $1}' release/hf/SHA256SUMS)
remote_sha() {  # LFS oid of the file on the Hub (= SHA-256), empty if absent
  curl -s -m 30 "https://huggingface.co/api/models/$REPO/tree/main" |
    python3 -c "import json,sys; print(next((x.get('lfs',{}).get('oid','') for x in json.load(sys.stdin) if x['path']=='$1'), ''))" 2>/dev/null
}
wait_net() { until curl -s -o /dev/null -m 10 https://huggingface.co/api/models/$REPO; do echo "$(date +%H:%M) network down, waiting"; sleep 30; done; }

for i in $(seq 1 $MAX); do
  wait_net
  if [[ $(remote_sha $FILE) == "$want" ]]; then echo "$(date +%H:%M) $FILE already on the Hub with the right SHA-256"; echo UPLOAD-DONE; exit 0; fi
  echo "=== attempt $i/$MAX $(date +%H:%M)"
  if (cd release/hf && $HF upload $REPO . . --repo-type model --commit-message "Welp-35B-A3B: 13.21 GB GGUF of Qwen3.6-35B-A3B"); then
    got=$(remote_sha $FILE)
    [[ $got == "$want" ]] && { echo "SHA-256 on the Hub matches: $got"; echo UPLOAD-DONE; exit 0; }
    echo "upload finished but the Hub has SHA-256 '$got', expected $want"
  fi
  pause=$(( i < 10 ? 60 * i : 600 ))
  echo "$(date +%H:%M) attempt $i failed, pausing ${pause}s"; sleep $pause
done
echo UPLOAD-GAVE-UP; exit 1
