#!/usr/bin/env bash
# After the IQ3 queue: Bonsai 2 27B on the same HumanEval harness (thinking off).
cd "$(dirname "$0")/.."
while pgrep -f "queue_iq3.sh" > /dev/null; do sleep 60; done
scripts/eval_coding.sh humaneval bonsai-27b
echo BONSAI-DONE
