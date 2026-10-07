#!/usr/bin/env bash
# Build every phase-1 variant from the BF16 GGUF. Skips files that already exist.
# Usage: scripts/quantize_all.sh [variant ...]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
Q=llama.cpp/build/bin/llama-quantize
SRC=models/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf
IMX=models/imatrix_unsloth.gguf_file
OUT=models/q
mkdir -p "$OUT" results/quant-logs

# non-expert tensors: attention, DeltaNet, shared expert, router
NONEXP=(attn_q attn_k attn_v attn_output attn_qkv attn_gate ssm_out ssm_alpha ssm_beta ffn_up_shexp ffn_gate_shexp ffn_down_shexp)

keep() { # keep() TYPE -> --tensor-type args for all non-expert tensors
  for t in "${NONEXP[@]}"; do printf -- '--tensor-type %s=%s ' "$t" "$1"; done
}

variant() {
  local name=$1; shift
  local f=$OUT/$name.gguf
  [[ -f $f ]] && { echo "skip $name"; return; }
  echo "== $name"
  # shellcheck disable=SC2068
  if env $@ > results/quant-logs/$name.log 2>&1; then :; else echo "FAILED $name"; rm -f "$f"; tail -5 results/quant-logs/$name.log; fi
  ls -la "$f" 2>/dev/null || true
}

ALL=(q4km iq2xxs q2k t0-absmax t1-fit t2-fit-q8 t3-fit-q4k t4-fit-q6k t5-fit-q6k-noimx t6-fit-q6k-imxsqrt t7-fit-q6k-imxkq t8-kq-q4k g0 g1 g2 g3 g4 tdown)
SEL=("${@:-${ALL[@]}}")
for v in "${SEL[@]}"; do
  case $v in
    # references and conventional low-bit baselines (same source, same imatrix)
    q4km)   variant $v $Q --imatrix $IMX $SRC $OUT/$v.gguf Q4_K_M ;;
    iq2xxs) variant $v $Q --imatrix $IMX $SRC $OUT/$v.gguf IQ2_XXS ;;
    q2k)    variant $v $Q --imatrix $IMX $SRC $OUT/$v.gguf Q2_K ;;
    # stock Prism ternary: absmax scale, default tensor mix
    t0-absmax) variant $v GGML_PTQ1_0_ABSMAX=1 $Q $SRC $OUT/$v.gguf PTQ1_0 ;;
    # MSE-fit ternary with imatrix, default tensor mix
    t1-fit) variant $v $Q --imatrix $IMX $SRC $OUT/$v.gguf PTQ1_0 ;;
    # ternary experts only, non-expert at Q8_0 / Q4_K
    t2-fit-q8)  variant $v $Q --imatrix $IMX $(keep q8_0) $SRC $OUT/$v.gguf PTQ1_0 ;;
    t3-fit-q4k) variant $v $Q --imatrix $IMX $(keep q4_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    # does the imatrix help the ternary fit?
    t4-fit-q6k)       variant $v $Q --imatrix $IMX $(keep q6_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    t5-fit-q6k-noimx) variant $v $Q $(keep q6_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    # softened imatrix weighting
    t6-fit-q6k-imxsqrt) variant $v GGML_PTQ1_0_IMX_POW=0.5 $Q --imatrix $IMX $(keep q6_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    t7-fit-q6k-imxkq)   variant $v GGML_PTQ1_0_IMX=kq $Q --imatrix $IMX $(keep q6_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    # phase 1c: kq weighting, rest Q4_K, plus targeted expert upgrades
    t8-kq-q4k) variant $v GGML_PTQ1_0_IMX=kq $Q --imatrix $IMX $(keep q4_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    g[0-4])    l=${v#g}; r=("[0-7]" "([89]|1[0-5])" "(1[6-9]|2[0-3])" "(2[4-9]|3[01])" "3[2-9]")
               variant $v GGML_PTQ1_0_IMX=kq $Q --imatrix $IMX --tensor-type "blk\.${r[$l]}\.ffn_(up|gate|down)_exps=q4_k" $(keep q4_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    tdown)     variant $v GGML_PTQ1_0_IMX=kq $Q --imatrix $IMX --tensor-type "ffn_down_exps=q2_k" $(keep q4_k) $SRC $OUT/$v.gguf PTQ1_0 ;;
    *) echo "unknown variant $v"; exit 1 ;;
  esac
done
