# Welp: ternary Qwen3.6-35B-A3B

Goal: a ternary (PTQ1_0, 1.75 bpw) Qwen3.6-35B-A3B that runs fully on a 16 GB RTX 4080 Super, as a recipe test before Flash-Next.

## Setup

Rebuild the engine: clone [professorpalmer/llama.cpp-ada-ternary](https://github.com/professorpalmer/llama.cpp-ada-ternary) branch `bonsai-q8-product` (PrismML fork plus the Bonsai Ada patches), `git am patches/0001-PTQ1_0-error-minimizing-ternary-fit-with-imatrix-wei.patch`, and build with CUDA (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`). The patch is only needed to make the files; running them works on any PrismML build. Python: `python -m venv .venv && .venv/bin/pip install torch numpy safetensors transformers tokenizers wasmtime`. Released files: [quaedra/Welp-35B-A3B-GGUF](https://huggingface.co/quaedra/Welp-35B-A3B-GGUF). Note: sizes in the experiment tables below are MiB/1000 (e.g. "9.4" = 9.82 GB, "8.2" = 8.64 GB, IQ2_XXS "9.1" = 9.50 GB).

- `llama.cpp/`: PrismML fork at `adfffbe` plus the 33 Bonsai Ada patches (`~/dev/bonsai-ada-surgery/patches`), branch `ternary-fit`, plus an error-minimizing PTQ1_0 quantizer in `ggml/src/ggml-quants.c` (`quantize_ptq1_0`).
  - default: per 128-weight group, choose the trit set and scale that minimize the weighted squared error
  - `GGML_PTQ1_0_ABSMAX=1`: original Prism absmax rounding (exact only for weights trained ternary)
  - `GGML_PTQ1_0_IMX=kq`: imatrix weight `qw * sqrt(sigma2 + x^2)` (best); `GGML_PTQ1_0_IMX_POW=p`: `qw^p`
- Source: Unsloth BF16 GGUF + `imatrix_unsloth.gguf_file` in `models/`.
- `scripts/quantize_all.sh <variant>`, `scripts/eval.sh <variant>`: build and measure variants.
- Eval: perplexity at ctx 2048, 40 chunks, on wikitext-2 test and on `data/code.txt` (10 CPython stdlib files); `llama-bench` pp512/tg128.

## Phase 1 results (2026-10-06, no training, post-training quantization only)

| variant | what | size GB | PPL wiki | PPL code | decode tok/s |
|---|---|---:|---:|---:|---:|
| q4km | Q4_K_M reference (does not fit in VRAM) | 21.2 | **5.62** | **1.67** | - |
| q2k | Q2_K + imatrix | 12.3 | 6.35 | 2.13 | 206 |
| iq2xxs | IQ2_XXS + imatrix | 9.1 | 7.83 | 2.65 | 212 |
| t0-absmax | stock Prism ternary, all weights | 8.2 | 7.2M | 19.8M | 214 |
| t1-fit | fitted ternary, all weights | 8.2 | 50.8 | 12.10 | 213 |
| t2-fit-q8 | ternary experts, rest Q8_0 | 8.9 | 8.81 | 2.82 | 156 |
| t3-fit-q4k | ternary experts, rest Q4_K | 8.2 | 8.86 | 2.83 | 187 |
| t4-fit-q6k | ternary experts, rest Q6_K, raw imatrix | 8.6 | 8.78 | 2.82 | 169 |
| t5-fit-q6k-noimx | same, no imatrix | 8.6 | 8.39 | 2.74 | 169 |
| t6-fit-q6k-imxsqrt | same, imatrix^0.5 | 8.6 | 8.40 | 2.74 | 169 |
| **t7-fit-q6k-imxkq** | same, k-quant imatrix weighting | 8.6 | **7.48** | **2.58** | 169 |

"rest" = attention, DeltaNet (`attn_qkv`, `attn_gate`, `ssm_out`, `ssm_alpha/beta`) and shared expert. Token embeddings and output head use the ftype default.

Findings:

1. Stock absmax ternary is useless on a model that was not trained ternary. The MSE fit makes it work.
2. Almost all damage of full ternary comes from the dense non-expert weights (50.8 -> 8.8 PPL). The 256 routed experts (32.2B of ~35B params) tolerate ternary well.
3. Non-expert precision above Q6_K buys nothing; Q4_K costs little and is faster.
4. Raw imatrix weighting hurts the ternary fit; k-quant style weighting helps a lot. t7 beats IQ2_XXS while 0.5 GB smaller.
5. Speed: ternary does not beat 2-bit on decode. Decode is set by the dense part (read every token), not the experts (8 of 256 per token).
6. Gap to Q4_K_M: +33% PPL on wiki, +55% on code. This is what phase 2 must close.

Expert routing (`results/expert_counts.txt`, Unsloth calibration, 837k tokens): layers 0-31 balanced; layers 32-39 skewed (rarest experts see 130-400 tokens, top 10% take 30-39% of traffic). Too few tokens for full-Hessian (GPTQ) statistics on rare deep experts.

## Phase 1c: targeted precision (2026-10-07, kq imatrix weighting, non-expert Q4_K)

| variant | change vs t8 | size GB | PPL wiki | PPL code |
|---|---|---:|---:|---:|
| t8-kq-q4k | ternary experts, rest Q4_K (decode 187 tok/s) | 8.2 | 7.55 | 2.60 |
| g0 | experts of layers 0-7 at Q4_K | 10.4 | 7.12 | 2.50 |
| g1 | layers 8-15 | 10.4 | 7.17 | 2.45 |
| g2 | layers 16-23 | 10.4 | 7.19 | 2.41 |
| g3 | layers 24-31 | 10.4 | 7.13 | 2.45 |
| g4 | layers 32-39 | 10.4 | 6.60 | 2.39 |
| tdown | all ffn_down_exps at Q2_K | 9.4 | 6.89 | 2.41 |

Layers 32-39 (the skewed-routing ones) are the most sensitive. Down-proj at Q2_K is the best gain per GB.

## Phase 2: GPTQ onto the ternary grid (`scripts/gptq_ternary.py`)

Layer by layer in PyTorch from the HF safetensors (`models/hf`): per-expert Hessians of gate/up inputs, batched GPTQ with the PTQ1_0 group grid and kq-weighted scale fit, down-proj Hessians recollected from the quantized gate/up (routing weight^2), quantized layer output feeds the next layer. Rare experts: Hessian + 512 pseudo-tokens of the layer mean. Calibration: half code (stdlib, llama.cpp sources/docs, no eval overlap), half wikitext-2 train, 2048-token sequences. `scripts/export_gguf.py` writes the trits into a copy of a PTQ1_0 GGUF (exact, via ggml's own packer). About 12 s per layer at 128 sequences on the 4080 Super.

| variant | base | calib seqs | size GB | PPL wiki | PPL code |
|---|---|---:|---:|---:|---:|
| rtn-a (control: same calib, no error feedback) | t8 | 128 | 8.2 | 7.50 | 2.59 |
| gptq-a | t8 | 128 | 8.2 | 7.28 | 2.48 |
| gptq-b | t8 | 256 | 8.2 | 7.01 | 2.44 |
| gptq-c-down2k (down stays Q2_K) | tdown | 128 | 9.4 | 6.67 | 2.30 |
| gptq-d | t8 | 512 | 8.2 | 7.09 | 2.40 |
| gptq-e-down2k | tdown | 512 | 9.4 | **6.52** | **2.24** |

Calibration gains level off at 256-512 sequences (gptq-b vs gptq-d is within noise on wiki).

## Phase 2b: scale tuning (`scripts/scale_tune.py`)

Trits fixed; per layer, learn a log-multiplier on every 128-weight group scale (Adam, lr 1e-2, cosine, 3 epochs, 256 seqs with a new calibration seed, 16 held out) so the routed-expert output matches the BF16 layer on the same input. Inputs come from the tuned previous layers. Holdout layer-output error drops 3-11% per layer. About 42 s per layer.

| variant | from | size GB | PPL wiki | PPL code |
|---|---|---:|---:|---:|
| st-d | gptq-d | 8.2 | 6.92 | 2.37 |
| st-e | gptq-e-down2k | 9.4 | **6.39** | **2.19** |

Kept builds (`models/q/`):

| file | size GB | PPL wiki | PPL code | decode tok/s |
|---|---:|---:|---:|---:|
| ternary-8.2g.gguf (st-d) | 8.2 | 6.92 | 2.37 | 186 |
| ternary-9.4g-down2k.gguf (st-e) | 9.4 | 6.39 | 2.19 | 197 |
| iq2xxs.gguf (baseline) | 9.1 | 7.83 | 2.65 | 212 |
| q4km.gguf (reference) | 21.2 | 5.62 | 1.67 | - |

Also kept: `t8-kq-q4k.gguf` and `tdown.gguf` (export bases), `models/gptq/{gptq-d,gptq-e-down2k,st-d,st-e}` (trits + scales). Superseded variants deleted; rebuild with `scripts/quantize_all.sh`, `scripts/run_gptq_set.sh`, `scripts/run_scale_set.sh`.

## Coding evals (2026-10-07)

HumanEval: 164 problems, thinking off, temperature 0, max 1024 tokens, `bonsai-ada-surgery/bench/humaneval_run.py`. Long-exact suite: `bonsai-ada-surgery/suite`, default plan (37 runs, thinking on, temperature 1.0), raw server. `scripts/eval_coding.sh`.

| model | size GB | fits fully in 16 GB | PPL wiki | PPL code | HumanEval | suite | decode tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| Q4_K_M | 21.2 | no (16 layers of experts on CPU) | 5.62 | 1.67 | 95.7% | - | 16 |
| Unsloth UD-IQ3_XXS | 13.2 | yes, up to 131k ctx (14.9 GB) | 5.87 | 1.90 | 93.9% | - | 169 |
| Bonsai 2 27B (dense, trained ternary) | 5.9 | yes, 262k | - | - | 91.5% | 17/37 (repo report) | 83 (no MTP) |
| ternary-9.4g-down2k | 9.4 | yes, 262k (13.3 GB) | 6.39 | 2.19 | 90.9% | 14/37 | 197 |
| iq2xxs | 9.1 | yes | 7.83 | 2.65 | 84.8% | 5/37 | 212 |
| ternary-8.2g | 8.2 | yes, 262k (12.2 GB) | 6.92 | 2.37 | 78.7% | - | 186 |

Notes: Q4_K_M with `--n-cpu-moe` needs `--no-op-offload` (op offload of CPU experts crashes with an illegal memory access on long prompts, stock Prism build too).

Long-exact suite, UD-IQ3_XXS: **21/37** (coding 2/12, computation 9/15, workspace 10/10).

Prefill (llama-bench, tokens/s, flash attention, RTX 4080 Super, mean of 3 runs; `results/prefill.txt`):

| model | pp512 | pp4096 | pp16384 |
|---|---:|---:|---:|
| ternary-9.4g-down2k | 5303 | 5442 | 5194 |
| ternary-8.2g | 5798 | 5958 | 5637 |
| iq2xxs | 5441 | 5706 | 5388 |
| ud-iq3xxs | 3486 | 4923 | 4909 |
| bonsai-27b | 2202 | 2271 | 2143 |
| q4km (16 layers of experts on CPU, no op offload) | 343 | 382 | - |

MTP on UD-IQ3_XXS (Unsloth MTP GGUF, 14.1 GB, `--spec-type draft-mtp --spec-draft-n-max 2`, q8_0 draft cache): served decode 212 tok/s vs 158 without (same 600-token code prompt, 66% draft acceptance), but only 64k context fits (15.7 GB) vs 131k. `scripts/serve.sh` (MODE=long|fast) and the `llama-qwen36` user service serve it on port 8080.

Full 262k context on 16 GB (prompt of 214k tokens, served; `scripts/ctx_test.sh`, `scripts/needle.py`):

| setup | VRAM | decode shallow | decode at 214k | prefill at 214k |
|---|---:|---:|---:|---:|
| UD-IQ3_XXS, q4_0 KV, -ub 256 | 14.8 GB | 155 | 107 | 2041 |
| UD-IQ3_XXS, tiered q8_0 KV (`--kv-vram-cells 120000`) | 14.9 GB | 156 | 12 | 2103 |
| UD-IQ3_XXS, K q8_0 + V q4_0 | fits | - | - | ~30 (no CUDA FA kernel, CPU fallback) |
| ternary-9.4g-down2k, q8_0 KV | 13.1 GB | 180 | 106 | 2511 |

Needle test, UD-IQ3_XXS with q4_0 KV at 214k: 3/3 (10%, 50%, 90% depth). This is the service default (`MODE=full`).
