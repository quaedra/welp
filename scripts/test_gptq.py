"""Synthetic check: GPTQ ternary lowers output error vs RTN; trits pack/unpack round-trip; export bytes dequantize back."""
import sys, os, ctypes, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_ternary import quantize_gptq, quantize_rtn, dequant, pack
from export_gguf import unpack, ptq1_bytes, ggml, PTQ1_0
torch.manual_seed(0)
E, R, C, N = 4, 256, 512, 4096
# correlated inputs with uneven channel scales, like real activations
A = torch.randn(C, C) / C**0.5 + torch.eye(C)
X = (torch.randn(E, N, C) @ A) * torch.logspace(-1, 0.5, C)
W = torch.randn(E, R, C) * 0.02
H = X.transpose(1, 2) @ X
for name, fn in (("rtn", lambda: quantize_rtn(W.cuda(), H.cuda())), ("gptq", lambda: quantize_gptq(W.cuda(), H.clone().cuda(), 0.01))):
    T, D = fn(); Q = dequant(T, D).cpu()
    err = ((X @ (W - Q).transpose(1, 2)) ** 2).sum() / ((X @ W.transpose(1, 2)) ** 2).sum()
    print(f"{name:5s} relative output MSE {err:.4f}  zeros {(T == 0).float().mean():.2f}")
assert torch.equal(unpack(pack(T.cpu())), T.cpu())
Tc, Dc = T.cpu()[0], D.cpu()[0].half()
b = ptq1_bytes(Tc, Dc)
ggml.ggml_get_type_traits.restype = ctypes.c_void_p
# dequantize through ggml: to_float is the 6th field of ggml_type_traits (name, blck_size, blck_size_interleave, type_size, is_quantized, to_float)
import numpy as np
tr = ggml.ggml_get_type_traits(PTQ1_0)
class Traits(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p)]
t = Traits.from_address(tr); assert t.name == b"ptq1_0", t.name
to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(t.to_float)
y = np.empty(Tc.numel(), dtype=np.float32); to_float(b.ctypes.data, y.ctypes.data, y.size)
ref = (Tc.float().view(R, -1, 128) * Dc.float().unsqueeze(-1)).view(-1).numpy()
print("export round-trip max abs diff", float(np.abs(y - ref).max()))
