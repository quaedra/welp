"""Block-wise GPTQ for the routed experts with ggml's own quantizers (any 256-block type: IQ2_S, IQ3_XXS, IQ4_XS, ...).

Per layer: run the BF16 layer on calibration hidden states, collect per-expert Hessians, then quantize each
expert matrix one 256-column block at a time with ggml (imatrix = diag of the expert Hessian) and push the
block's rounding error into the columns not yet quantized (lazy-batch GPTQ, Cholesky form). The tensor
types come from a base GGUF; the output is the exact ggml bytes of each expert tensor, written into a copy
of the base by export_ggml.py. Quantized layer output feeds the next layer.
"""
import argparse, ctypes, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
from gptq_ternary import Weights, calib_tokens, parse_layers

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{root}/llama.cpp/gguf-py")
from gguf import GGUFReader

QK = 256
ggml = ctypes.CDLL(f"{root}/llama.cpp/build/bin/libggml-base.so")
ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
ggml.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                     ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
ggml.ggml_row_size.restype = ctypes.c_size_t
ggml.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
ggml.ggml_get_type_traits.restype = ctypes.c_void_p


class Traits(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p)]


_to_float = {}
def to_float(qtype):
    if qtype not in _to_float:
        t = Traits.from_address(ggml.ggml_get_type_traits(qtype))
        _to_float[qtype] = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(t.to_float)
    return _to_float[qtype]


POOL = ThreadPoolExecutor(12)


def quant_block(qtype, W, imx):
    """W: float32 [E, R, QK] (CPU), imx: float32 [E, QK]. Returns (bytes [E, R, bs] uint8, dequant [E, R, QK])."""
    E, R, _ = W.shape
    bs = ggml.ggml_row_size(qtype, QK)
    W = np.ascontiguousarray(W, dtype=np.float32)
    imx = np.ascontiguousarray(imx, dtype=np.float32)
    out = np.empty((E, R, bs), np.uint8)
    deq = np.empty((E, R, QK), np.float32)
    f = to_float(qtype)

    def job(e):
        ggml.ggml_quantize_chunk(qtype, W[e].ctypes.data, out[e].ctypes.data, 0, R, QK, imx[e].ctypes.data)
        f(out[e].ctypes.data, deq[e].ctypes.data, R * QK)
    list(POOL.map(job, range(E)))
    return out, deq


def gptq_ggml(W, H, qtype, damp, mode):
    """W: [E,R,C] float32 cuda, H: [E,C,C] cuda (modified). Returns bytes [E, R, C/QK, bs] uint8 and dequant W [E,R,C]."""
    E, R, C = W.shape
    W = W.clone()
    hdiag = torch.diagonal(H, dim1=-2, dim2=-1).clone()
    idx = torch.arange(C, device=W.device)
    dead = hdiag <= 0
    H[:, idx, idx] += damp * hdiag.mean(-1, keepdim=True).clamp_min(1e-8) + dead.float()
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    imx_all = (hdiag / hdiag.mean(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-6).cpu().numpy()
    blocks, Q = [], torch.empty_like(W)
    for b0 in range(0, C, QK):
        b1 = b0 + QK
        byt, deq = quant_block(qtype, W[:, :, b0:b1].cpu().numpy(), imx_all[:, b0:b1])
        q = torch.from_numpy(deq).to(W.device)
        Q[:, :, b0:b1] = q
        blocks.append(byt)
        if mode == "gptq" and b1 < C:
            # Err = (W_b - Q_b) U_bb^-1 ; W_rest -= Err U_b,rest
            delta = W[:, :, b0:b1] - q
            err = torch.linalg.solve_triangular(U[:, b0:b1, b0:b1], delta, upper=True, left=False)
            W[:, :, b1:] -= torch.bmm(err, U[:, b0:b1, b1:])
    return np.stack(blocks, axis=2), Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="models/hf")
    ap.add_argument("--base", default="models/q/ud-iq3xxs.gguf", help="GGUF whose expert tensor types are used")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["gptq", "rtn"], default="gptq")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--nseq", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--code-frac", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--prior", type=float, default=512)
    ap.add_argument("--echunk", type=int, default=32)
    args = ap.parse_args()

    os.chdir(root)
    out_dir = f"models/gptq/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    cfg = AutoConfig.from_pretrained(args.hf).text_config
    cfg._attn_implementation = "sdpa"
    E, FF, HID = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    wts = Weights(args.hf)
    layers = parse_layers(args.layers, cfg.num_hidden_layers)
    assert layers[0] == 0
    types = {t.name: int(t.tensor_type) for t in GGUFReader(args.base).tensors if "_exps" in t.name}

    tok = Tokenizer.from_file(f"{args.hf}/tokenizer.json")
    ids = calib_tokens(tok, root, args.nseq, args.seqlen, args.code_frac)
    emb = wts.get(["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    X = F.embedding(ids, emb).to(torch.bfloat16)
    del emb
    N, S = X.shape[:2]
    assert N % args.bs == 0
    rope = M.Qwen3_5MoeTextRotaryEmbedding(config=cfg).to(dev)
    pos = torch.arange(S, device=dev).view(1, 1, -1).expand(3, args.bs, -1)
    with torch.no_grad():
        pe = rope(torch.zeros(1, device=dev, dtype=torch.bfloat16), pos)

    def shrink(H, cnt):
        return H + args.prior * H.sum(0) / cnt.sum()

    def quantize_tensor(Wall, Hall, qtype):
        Eall, R, C = Wall.shape
        out = np.empty((Eall, R, C // QK, ggml.ggml_row_size(qtype, QK)), np.uint8)
        for e0 in range(0, Eall, args.echunk):
            e1 = e0 + args.echunk
            byt, Q = gptq_ggml(Wall[e0:e1].float(), Hall[e0:e1].clone(), qtype, args.damp, args.mode)
            out[e0:e1] = byt
            Wall[e0:e1] = Q.to(Wall.dtype)
        return out

    for li in layers:
        t0 = time.time()
        layer = M.Qwen3_5MoeDecoderLayer(cfg, li)
        missing, unexpected = layer.load_state_dict(wts.layer(li), strict=False)
        assert not missing and not unexpected
        layer = layer.to(dev, torch.bfloat16).eval()
        moe = layer.mlp

        Hgu = torch.zeros(E, HID, HID, device=dev)
        cnt = torch.zeros(E, device=dev)
        Mx = torch.empty(N, S, HID, dtype=torch.bfloat16, pin_memory=True)
        TI = torch.empty(N, S, cfg.num_experts_per_tok, dtype=torch.int16, device=dev)
        TW = torch.empty(N, S, cfg.num_experts_per_tok, dtype=torch.float32, device=dev)
        with torch.no_grad():
            for b in range(0, N, args.bs):
                x = X[b:b + args.bs].to(dev)
                h = layer.input_layernorm(x)
                if layer.block_type == "linear_attention":
                    h = layer.linear_attn(hidden_states=h)
                else:
                    h, _ = layer.self_attn(hidden_states=h, position_embeddings=pe, attention_mask=None)
                r = x + h
                m = layer.post_attention_layernorm(r)
                X[b:b + args.bs] = r.cpu()
                Mx[b:b + args.bs].copy_(m)
                flat = m.view(-1, HID)
                _, sc, ti = moe.gate(flat)
                TI[b:b + args.bs] = ti.view(x.shape[0], S, -1).to(torch.int16)
                TW[b:b + args.bs] = sc.view(x.shape[0], S, -1).float()
                xf = flat.float()
                for e in torch.unique(ti).tolist():
                    xe = xf[(ti == e).any(-1)]
                    Hgu[e].addmm_(xe.T, xe)
                    cnt[e] += xe.shape[0]
        Hgu = shrink(Hgu, cnt)

        WGU = moe.experts.gate_up_proj.data  # [E, 2FF, HID]; rows [:FF] gate, [FF:] up
        res = {}
        for part, sl in (("gate", slice(0, FF)), ("up", slice(FF, 2 * FF))):
            Wp = WGU[:, sl].clone()
            res[part] = quantize_tensor(Wp, Hgu, types[f"blk.{li}.ffn_{part}_exps.weight"])
            WGU[:, sl] = Wp
        del Hgu

        Hd = torch.zeros(E, FF, FF, device=dev)
        cntd = torch.zeros(E, device=dev)
        with torch.no_grad():
            for b in range(0, N, args.bs):
                flat = Mx[b:b + args.bs].to(dev).view(-1, HID)
                ti = TI[b:b + args.bs].view(-1, TI.shape[-1]).long()
                tw = TW[b:b + args.bs].view(-1, TW.shape[-1])
                for e in torch.unique(ti).tolist():
                    tok_i, k_i = torch.where(ti == e)
                    g, u = F.linear(flat[tok_i], WGU[e]).chunk(2, dim=-1)
                    z = (F.silu(g) * u).float() * tw[tok_i, k_i].unsqueeze(-1)
                    Hd[e].addmm_(z.T, z)
                    cntd[e] += z.shape[0]
        Hd = shrink(Hd, cntd)
        res["down"] = quantize_tensor(moe.experts.down_proj.data, Hd, types[f"blk.{li}.ffn_down_exps.weight"])
        del Hd

        with torch.no_grad():
            for b in range(0, N, args.bs):
                m = Mx[b:b + args.bs].to(dev)
                X[b:b + args.bs] = (X[b:b + args.bs].to(dev) + moe(m)).cpu()

        np.savez(f"{out_dir}/layer_{li:02d}.npz", **res)
        print(f"layer {li:2d} {time.time() - t0:6.1f}s", flush=True)
        del layer, moe, Mx, TI, TW
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
