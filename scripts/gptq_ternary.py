"""Layer-by-layer GPTQ for the routed experts of Qwen3.6-35B-A3B onto the PTQ1_0 grid
(ternary, one scale per 128 input weights).

For each decoder layer: run the BF16 layer on the calibration hidden states, collect per-expert
Hessians of the gate/up inputs, quantize gate/up, recollect down-proj Hessians from the quantized
gate/up outputs (weighted by routing weight^2), quantize down, then pass the quantized layer output
on as the next layer's input. Rare experts get their Hessian shrunk toward the layer-wide mean.

Output per layer: models/gptq/<tag>/layer_XX.pt with packed trits and fp16 scales for
gate [E,512,2048], up [E,512,2048], down [E,2048,512].
"""
import argparse, json, math, os, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

G = 128  # PTQ1_0 group size


def parse_layers(s, n):
    if s == "all":
        return list(range(n))
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# ---------------------------------------------------------------- calibration
def calib_tokens(tok, root, nseq, seqlen, code_frac, seed=0):
    rnd = random.Random(seed)
    streams = {}
    for name in ("code", "wiki"):
        txt = open(f"{root}/data/calib_{name}.txt", encoding="utf-8").read()
        streams[name] = tok.encode(txt).ids
    seqs = []
    for i in range(nseq):
        s = streams["code" if i < round(nseq * code_frac) else "wiki"]
        j = rnd.randrange(0, len(s) - seqlen)
        seqs.append(s[j:j + seqlen])
    return torch.tensor(seqs, dtype=torch.long)


# ---------------------------------------------------------------- weights
class Weights:
    def __init__(self, hf):
        self.hf = hf
        self.map = json.load(open(f"{hf}/model.safetensors.index.json"))["weight_map"]

    def get(self, keys):
        by_file = {}
        for k in keys:
            by_file.setdefault(self.map[k], []).append(k)
        out = {}
        for fn, ks in by_file.items():
            with safe_open(f"{self.hf}/{fn}", framework="pt") as f:
                for k in ks:
                    out[k] = f.get_tensor(k)
        return out

    def layer(self, i):
        pre = f"model.language_model.layers.{i}."
        sd = self.get([k for k in self.map if k.startswith(pre)])
        return {k[len(pre):]: v for k, v in sd.items()}


# ---------------------------------------------------------------- ternary fit
def fit_scale(w, a, lloyd=4):
    """w, a: [..., G]. Scale d minimizing sum a (w - d t)^2 over trits t. Returns d [...]."""
    aw = w.abs()
    s, idx = aw.sort(dim=-1, descending=True)
    sa = a.gather(-1, idx)
    s1 = (sa * s).cumsum(-1)
    s0 = sa.cumsum(-1).clamp_min(1e-30)
    k = (s1 * s1 / s0).argmax(-1, keepdim=True)
    d = (s1.gather(-1, k) / s0.gather(-1, k)).squeeze(-1)
    for _ in range(lloyd):
        m = (aw > 0.5 * d.unsqueeze(-1)).float() * a
        n0 = m.sum(-1)
        d = torch.where(n0 > 0, (m * aw).sum(-1) / n0.clamp_min(1e-30), d)
    return d


def trit(w, d):
    return torch.where(w.abs() > 0.5 * d, w.sign(), torch.zeros_like(w))


def imx_weight(w, hdiag):
    """k-quant style weight: imatrix * sqrt(sigma2 + w^2); sigma2 over each row. w: [E,R,C], hdiag: [E,C]."""
    sigma2 = 2 * w.pow(2).mean(-1, keepdim=True)
    return hdiag.unsqueeze(1) * (sigma2 + w * w).sqrt()


def quantize_rtn(W, H):
    """W: [E,R,C] float32, H: [E,C,C]. Returns trits int8 [E,R,C], scales [E,R,C/G]."""
    E, R, C = W.shape
    a = imx_weight(W, torch.diagonal(H, dim1=-2, dim2=-1)).view(E, R, C // G, G)
    d = fit_scale(W.view(E, R, C // G, G), a)
    t = trit(W.view(E, R, C // G, G), d.unsqueeze(-1))
    return t.view(E, R, C).to(torch.int8), d


def quantize_gptq(W, H, damp):
    """Batched GPTQ over experts with the ternary group grid. W: [E,R,C], H: [E,C,C] (modified)."""
    E, R, C = W.shape
    W = W.clone()
    hdiag0 = torch.diagonal(H, dim1=-2, dim2=-1).clone()
    dead = hdiag0 <= 0
    idx = torch.arange(C, device=W.device)
    H[:, idx, idx] += damp * hdiag0.mean(-1, keepdim=True).clamp_min(1e-8) + dead.float()
    W.masked_fill_(dead.unsqueeze(1), 0)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    T = torch.zeros(E, R, C, dtype=torch.int8, device=W.device)
    D = torch.zeros(E, R, C // G, device=W.device)
    for g0 in range(0, C, G):
        g1 = g0 + G
        W1 = W[:, :, g0:g1].clone()
        Err = torch.zeros_like(W1)
        Hi = Hinv[:, g0:g1, g0:g1]
        sigma2 = 2 * W.pow(2).mean(-1, keepdim=True)
        a = hdiag0[:, None, g0:g1] * (sigma2 + W1 * W1).sqrt()
        d = fit_scale(W1, a)
        D[:, :, g0 // G] = d
        for i in range(G):
            w = W1[:, :, i]
            t = trit(w, d)
            q = d * t
            e = (w - q) / Hi[:, i, i].unsqueeze(-1)
            W1[:, :, i:] -= e.unsqueeze(-1) * Hi[:, i, i:].unsqueeze(1)
            Err[:, :, i] = e
            T[:, :, g0 + i] = t.to(torch.int8)
        W[:, :, g1:] -= torch.bmm(Err, Hinv[:, g0:g1, g1:])
    return T, D


def dequant(T, D):
    E, R, C = T.shape
    return (T.float().view(E, R, C // G, G) * D.unsqueeze(-1)).view(E, R, C)


def pack(T):  # int8 trits -> uint8, 4 per byte
    u = (T + 1).to(torch.uint8).view(*T.shape[:-1], -1, 4)
    return u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="models/hf")
    ap.add_argument("--tag", default="gptq")
    ap.add_argument("--mode", choices=["gptq", "rtn"], default="gptq")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--nseq", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--code-frac", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--prior", type=float, default=512, help="pseudo-tokens of layer-mean Hessian added to each expert")
    ap.add_argument("--echunk", type=int, default=32, help="experts per batched GPTQ call")
    ap.add_argument("--keep-down", action="store_true", help="leave down_proj unquantized (it gets a higher-precision type at export)")
    ap.add_argument("--save-inputs", action="store_true", help="checkpoint hidden states after each layer for --resume")
    ap.add_argument("--resume", action="store_true", help="start at the first missing layer, reloading its saved input")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    out_dir = f"models/gptq/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False

    cfg = AutoConfig.from_pretrained(args.hf).text_config
    cfg._attn_implementation = "sdpa"
    E, FF, HID = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    wts = Weights(args.hf)
    layers = parse_layers(args.layers, cfg.num_hidden_layers)

    inp_path = f"{out_dir}/inputs.pt"  # hidden states entering the next layer to process
    if args.resume and os.path.exists(inp_path):
        state = torch.load(inp_path)
        X, start = state["X"], state["next_layer"]
        layers = [l for l in layers if l >= start]
        print(f"resume at layer {start}")
    else:
        tok = Tokenizer.from_file(f"{args.hf}/tokenizer.json")
        ids = calib_tokens(tok, root, args.nseq, args.seqlen, args.code_frac)
        emb = wts.get(["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
        X = F.embedding(ids, emb).to(torch.bfloat16)  # [N,S,HID] on CPU
        del emb
        assert layers[0] == 0, "start from layer 0 or use --resume"
    N, S = X.shape[0], X.shape[1]
    assert N % args.bs == 0

    rope = M.Qwen3_5MoeTextRotaryEmbedding(config=cfg).to(dev)
    pos = torch.arange(S, device=dev).view(1, 1, -1).expand(3, args.bs, -1)
    with torch.no_grad():
        pe = rope(torch.zeros(1, device=dev, dtype=torch.bfloat16), pos)

    for li in layers:
        t0 = time.time()
        layer = M.Qwen3_5MoeDecoderLayer(cfg, li)
        missing, unexpected = layer.load_state_dict(wts.layer(li), strict=False)
        assert not unexpected and not missing, (missing, unexpected)
        layer = layer.to(dev, torch.bfloat16).eval()
        moe = layer.mlp

        # pass 1: token mixer, router, gate/up Hessians
        Hgu = torch.zeros(E, HID, HID, device=dev)
        cnt = torch.zeros(E, device=dev)
        Mx = torch.empty(N, S, HID, dtype=torch.bfloat16, pin_memory=True)  # MoE input (CPU, pinned)
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
                X[b:b + args.bs] = r.cpu()  # X now holds the residual after the token mixer
                Mx[b:b + args.bs].copy_(m, non_blocking=True)
                flat = m.view(-1, HID)
                _, sc, ti = moe.gate(flat)
                TI[b:b + args.bs] = ti.view(x.shape[0], S, -1).to(torch.int16)
                TW[b:b + args.bs] = sc.view(x.shape[0], S, -1).float()
                xf = flat.float()
                for e in torch.unique(ti).tolist():
                    sel = (ti == e).any(-1)
                    xe = xf[sel]
                    Hgu[e].addmm_(xe.T, xe)
                    cnt[e] += xe.shape[0]

        def shrink(H, cnt):
            mean = H.sum(0) / cnt.sum()
            return H + args.prior * mean

        Hgu = shrink(Hgu, cnt)
        WGU = moe.experts.gate_up_proj.data  # [E, 2FF, HID] bf16
        Tg = torch.empty(E, 2 * FF, HID, dtype=torch.int8)
        Dg = torch.empty(E, 2 * FF, HID // G)
        for e0 in range(0, E, args.echunk):
            e1 = e0 + args.echunk
            W = WGU[e0:e1].float()
            H = Hgu[e0:e1].clone()
            T, D = quantize_gptq(W, H, args.damp) if args.mode == "gptq" else quantize_rtn(W, H)
            Tg[e0:e1], Dg[e0:e1] = T.cpu(), D.cpu()
            WGU[e0:e1] = dequant(T, D).to(WGU.dtype)
        del Hgu

        Td = Dd = None
        if not args.keep_down:
            # pass 2: down-proj Hessians from quantized gate/up, weighted by routing weight^2
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
            WD = moe.experts.down_proj.data  # [E, HID, FF]
            Td = torch.empty(E, HID, FF, dtype=torch.int8)
            Dd = torch.empty(E, HID, FF // G)
            for e0 in range(0, E, args.echunk):
                e1 = e0 + args.echunk
                W = WD[e0:e1].float()
                H = Hd[e0:e1].clone()
                T, D = quantize_gptq(W, H, args.damp) if args.mode == "gptq" else quantize_rtn(W, H)
                Td[e0:e1], Dd[e0:e1] = T.cpu(), D.cpu()
                WD[e0:e1] = dequant(T, D).to(WD.dtype)
            del Hd

        # pass 3: layer output with quantized experts -> next layer input
        with torch.no_grad():
            for b in range(0, N, args.bs):
                m = Mx[b:b + args.bs].to(dev)
                X[b:b + args.bs] = (X[b:b + args.bs].to(dev) + moe(m)).cpu()

        torch.save({
            "gate_t": pack(Tg[:, :FF]), "gate_d": Dg[:, :FF].half(),
            "up_t": pack(Tg[:, FF:]), "up_d": Dg[:, FF:].half(),
            **({"down_t": pack(Td), "down_d": Dd.half()} if Td is not None else {}),
            "count": cnt.cpu(), "mode": args.mode,
        }, f"{out_dir}/layer_{li:02d}.pt")
        if args.save_inputs:
            torch.save({"X": X, "next_layer": li + 1}, inp_path)
        c = cnt.sort().values
        print(f"layer {li:2d} {layer.block_type:17s} {time.time() - t0:6.1f}s  tokens/expert min {c[0]:.0f} median {c[E // 2]:.0f}", flush=True)
        del layer, moe, Mx, TI, TW
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
