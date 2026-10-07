"""Block-wise scale tuning for GPTQ ternary experts.

Trits stay fixed. Per layer, learn a multiplier exp(s) on every PTQ1_0 group scale so that the
routed-expert output of the quantized layer matches the BF16 layer on the same inputs. Inputs come
from the already tuned previous layers. Reads models/gptq/<src>/layer_XX.pt, writes the same format
to models/gptq/<tag>/ (export with scripts/export_gguf.py).
"""
import argparse, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
from gptq_ternary import Weights, calib_tokens, pack, parse_layers, G
from export_gguf import unpack


def routing_plan(ti, tw, E):
    """Group (token, slot) pairs by expert. Returns token idx, weights (sorted by expert) and per-expert bounds."""
    flat_e = ti.reshape(-1).long()
    order = torch.argsort(flat_e, stable=True)
    k = ti.shape[-1]
    tok = order // k
    w = tw.reshape(-1)[order]
    bounds = torch.zeros(E + 1, dtype=torch.long, device=ti.device)
    bounds[1:] = torch.bincount(flat_e, minlength=E).cumsum(0)
    return tok, w, bounds.tolist()


def routed(flat, plan, gu, down):
    """Sum of routed expert outputs. gu(e) -> [2FF,HID], down(e) -> [HID,FF] (bf16)."""
    tok, w, b = plan
    out = torch.zeros(flat.shape[0], flat.shape[1], dtype=torch.float32, device=flat.device)
    for e in range(len(b) - 1):
        if b[e] == b[e + 1]:
            continue
        t = tok[b[e]:b[e + 1]]
        g, u = F.linear(flat[t], gu(e)).chunk(2, dim=-1)
        y = F.linear(F.silu(g) * u, down(e))
        out.index_add_(0, t, y.float() * w[b[e]:b[e + 1], None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="models/hf")
    ap.add_argument("--src", required=True, help="GPTQ tag with trits")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--nseq", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--code-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=int, default=16, help="sequences kept out of training for the loss report")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    src, out_dir = f"models/gptq/{args.src}", f"models/gptq/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    cfg = AutoConfig.from_pretrained(args.hf).text_config
    cfg._attn_implementation = "sdpa"
    E, FF, HID, K = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts_per_tok
    wts = Weights(args.hf)
    layers = parse_layers(args.layers, cfg.num_hidden_layers)
    assert layers[0] == 0

    tok = Tokenizer.from_file(f"{args.hf}/tokenizer.json")
    ids = calib_tokens(tok, root, args.nseq, args.seqlen, args.code_frac, seed=args.seed)
    ids = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed))]  # mix code/wiki across batches
    emb = wts.get(["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    X = F.embedding(ids, emb).to(torch.bfloat16)
    del emb
    N, S = X.shape[:2]
    assert N % args.bs == 0 and args.holdout % args.bs == 0
    train_b = list(range(args.holdout, N, args.bs))
    hold_b = list(range(0, args.holdout, args.bs))

    rope = M.Qwen3_5MoeTextRotaryEmbedding(config=cfg).to(dev)
    pos = torch.arange(S, device=dev).view(1, 1, -1).expand(3, args.bs, -1)
    with torch.no_grad():
        pe = rope(torch.zeros(1, device=dev, dtype=torch.bfloat16), pos)

    for li in layers:
        t0 = time.time()
        layer = M.Qwen3_5MoeDecoderLayer(cfg, li)
        missing, unexpected = layer.load_state_dict(wts.layer(li), strict=False)
        assert not missing and not unexpected
        layer = layer.to(dev, torch.bfloat16).eval()
        moe = layer.mlp
        q = torch.load(f"{src}/layer_{li:02d}.pt")
        has_down = "down_t" in q

        # token mixer, MoE inputs, routing; X becomes the residual
        Mx = torch.empty(N, S, HID, dtype=torch.bfloat16, pin_memory=True)
        Yref = torch.empty(N, S, HID, dtype=torch.bfloat16, pin_memory=True)  # BF16 routed-expert output
        plans = {}
        WGU0, WD0 = moe.experts.gate_up_proj.data, moe.experts.down_proj.data
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
                plans[b] = routing_plan(ti, sc.float(), E)
                Yref[b:b + args.bs].copy_(routed(flat, plans[b], lambda e: WGU0[e], lambda e: WD0[e]).view(m.shape).to(torch.bfloat16))

        # quantized experts with learnable log-multipliers on the group scales
        Tgu = torch.cat([unpack(q["gate_t"]), unpack(q["up_t"])], 1).to(dev, torch.bfloat16)  # [E,2FF,HID]
        Dgu0 = torch.cat([q["gate_d"], q["up_d"]], 1).float().to(dev)                 # [E,2FF,HID/G]
        sgu = torch.zeros_like(Dgu0, requires_grad=True)
        params = [sgu]
        if has_down:
            Td = unpack(q["down_t"]).to(dev, torch.bfloat16)
            Dd0 = q["down_d"].float().to(dev)
            sd = torch.zeros_like(Dd0, requires_grad=True)
            params.append(sd)

        def gu(e):
            return Tgu[e] * (Dgu0[e] * sgu[e].exp()).to(torch.bfloat16).repeat_interleave(G, -1)

        def down(e):
            if not has_down:
                return WD0[e]
            return Td[e] * (Dd0[e] * sd[e].exp()).to(torch.bfloat16).repeat_interleave(G, -1)

        def eval_loss(bs):
            num = den = 0.0
            with torch.no_grad():
                for b in bs:
                    y = routed(Mx[b:b + args.bs].to(dev).view(-1, HID), plans[b], gu, down)
                    ref = Yref[b:b + args.bs].to(dev).view(-1, HID).float()
                    num += (y - ref).pow(2).sum().item(); den += ref.pow(2).sum().item()
            return num / den

        before = eval_loss(hold_b)
        opt = torch.optim.Adam(params, lr=args.lr)
        steps = args.epochs * len(train_b)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        g = torch.Generator().manual_seed(li)
        for ep in range(args.epochs):
            for i in torch.randperm(len(train_b), generator=g).tolist():
                b = train_b[i]
                y = routed(Mx[b:b + args.bs].to(dev).view(-1, HID), plans[b], gu, down)
                ref = Yref[b:b + args.bs].to(dev).view(-1, HID).float()
                loss = (y - ref).pow(2).mean() / ref.pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step(); sched.step()
        after = eval_loss(hold_b)

        # write tuned weights into the layer, pass output on
        with torch.no_grad():
            Dgu = Dgu0 * sgu.exp()
            for e in range(E):
                WGU0[e] = gu(e)
                if has_down:
                    WD0[e] = down(e)
            for b in range(0, N, args.bs):
                m = Mx[b:b + args.bs].to(dev)
                X[b:b + args.bs] = (X[b:b + args.bs].to(dev) + moe(m)).cpu()

        outq = {"gate_t": q["gate_t"], "up_t": q["up_t"],
                "gate_d": Dgu[:, :FF].half().cpu(), "up_d": Dgu[:, FF:].half().cpu(),
                "count": q["count"], "mode": "gptq+scale"}
        if has_down:
            outq.update(down_t=q["down_t"], down_d=(Dd0 * sd.exp()).detach().half().cpu())
        torch.save(outq, f"{out_dir}/layer_{li:02d}.pt")
        print(f"layer {li:2d} {time.time() - t0:6.1f}s  holdout rel err {before:.4f} -> {after:.4f}", flush=True)
        del layer, moe, Mx, Yref, plans, Tgu, opt, params
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
