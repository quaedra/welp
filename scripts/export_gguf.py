"""Write GPTQ ternary experts into a copy of a PTQ1_0 GGUF (same tensor types and sizes).

Usage: export_gguf.py BASE.gguf GPTQ_DIR OUT.gguf [--layers 0-39]
BASE must have PTQ1_0 ffn_{gate,up,down}_exps (e.g. models/q/t8-kq-q4k.gguf). Packing uses ggml's
own quantize_ptq1_0 in absmax mode, which is exact for values that are already d * {-1,0,1}.
"""
import argparse, ctypes, os, shutil, sys
os.environ["GGML_PTQ1_0_ABSMAX"] = "1"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{root}/llama.cpp/gguf-py")
import numpy as np
import torch
from gguf import GGUFReader, GGMLQuantizationType

ggml = ctypes.CDLL(f"{root}/llama.cpp/build/bin/libggml-base.so")
ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
ggml.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                     ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
PTQ1_0 = int(GGMLQuantizationType.PTQ1_0)
G = 128


def unpack(p):  # uint8 [..., C/4] -> int8 trits [..., C]
    t = torch.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).flatten(-2)
    return t.to(torch.int8) - 1


def ptq1_bytes(T, D):
    """T int8 [rows, C], D fp16 [rows, C/G] -> PTQ1_0 bytes."""
    rows, C = T.shape
    x = (T.float().view(rows, C // G, G) * D.float().unsqueeze(-1)).view(rows, C).contiguous().numpy()
    out = np.empty(rows * (C // G) * 28, dtype=np.uint8)
    n = ggml.ggml_quantize_chunk(PTQ1_0, x.ctypes.data, out.ctypes.data, 0, rows, C, None)
    assert n == out.nbytes, (n, out.nbytes)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("gptq_dir"); ap.add_argument("out")
    ap.add_argument("--layers", default=None)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.gptq_dir) if f.startswith("layer_"))
    if a.layers:
        lo, _, hi = a.layers.partition("-")
        keep = set(range(int(lo), int(hi or lo) + 1))
        files = [f for f in files if int(f[6:8]) in keep]
    shutil.copyfile(a.base, a.out)
    r = GGUFReader(a.out)
    tensors = {t.name: t for t in r.tensors}
    with open(a.out, "r+b") as f:
        for fn in files:
            li = int(fn[6:8])
            st = torch.load(f"{a.gptq_dir}/{fn}")
            for part, gname in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"), ("down", "ffn_down_exps")):
                if f"{part}_t" not in st:  # not quantized by GPTQ (--keep-down); base keeps its own type
                    continue
                t = tensors[f"blk.{li}.{gname}.weight"]
                assert t.tensor_type == GGMLQuantizationType.PTQ1_0, (t.name, t.tensor_type)
                T = unpack(st[f"{part}_t"])            # [E, R, C]
                E, R_, C = T.shape
                b = ptq1_bytes(T.reshape(E * R_, C), st[f"{part}_d"].reshape(E * R_, C // G))
                assert b.nbytes == t.n_bytes, (t.name, b.nbytes, t.n_bytes)
                f.seek(t.data_offset); f.write(b.tobytes())
            print(f"layer {li:2d} written", flush=True)


if __name__ == "__main__":
    main()
