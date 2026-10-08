"""Write gptq_ggml.py expert bytes into a copy of the base GGUF (same tensor types and sizes).

Usage: export_ggml.py BASE.gguf GPTQ_DIR OUT.gguf
"""
import os, shutil, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{root}/llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader

base, src, out = sys.argv[1:4]
files = sorted(f for f in os.listdir(src) if f.startswith("layer_") and f.endswith(".npz"))
shutil.copyfile(base, out)
tensors = {t.name: t for t in GGUFReader(out).tensors}
with open(out, "r+b") as f:
    for fn in files:
        li = int(fn[6:8])
        z = np.load(f"{src}/{fn}")
        for part in ("gate", "up", "down"):
            t = tensors[f"blk.{li}.ffn_{part}_exps.weight"]
            b = np.ascontiguousarray(z[part])
            assert b.nbytes == t.n_bytes, (t.name, b.nbytes, t.n_bytes)
            f.seek(t.data_offset)
            f.write(b.tobytes())
        print(f"layer {li:2d} written", flush=True)
