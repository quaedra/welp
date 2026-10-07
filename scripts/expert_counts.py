"""Per-expert calibration token counts from a GGUF imatrix (the *.counts tensors)."""
import struct, sys, statistics as st, array

def read_gguf(path):
    f = open(path, 'rb')
    r = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
    assert f.read(4) == b'GGUF'
    ver, = r('I'); nt, nkv = r('QQ')
    def s(): n, = r('Q'); return f.read(n).decode()
    sz = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def val(t):
        if t in sz: return r(sz[t])[0]
        if t == 8: return s()
        et, n = r('I')[0], r('Q')[0]; return [val(et) for _ in range(n)]
    kv = {}
    for _ in range(nkv):
        k = s(); kv[k] = val(r('I')[0])
    infos = []
    for _ in range(nt):
        name = s(); nd, = r('I'); dims = r('Q' * nd); ty, = r('I'); off, = r('Q')
        infos.append((name, dims, ty, off))
    align = kv.get('general.alignment', 32)
    base = (f.tell() + align - 1) // align * align
    return f, kv, infos, base

f, kv, infos, base = read_gguf(sys.argv[1])
print({k: v for k, v in kv.items() if 'imatrix' in k and not isinstance(v, list)})
for name, dims, ty, off in infos:
    if not (name.endswith('.counts') and 'ffn_up_exps' in name): continue
    assert ty == 0  # f32
    n = 1
    for d in dims: n *= d
    f.seek(base + off); a = array.array('f'); a.frombytes(f.read(4 * n))
    c = sorted(a); tot = sum(c)
    layer = int(name.split('.')[1])
    print(f"layer {layer:2d}: tokens/expert min {c[0]:7.0f} p10 {c[len(c)//10]:7.0f} median {st.median(c):7.0f} max {c[-1]:8.0f}  "
          f"top10% share {sum(c[-len(c)//10:])/tot:5.1%}  <100 tok: {sum(x < 100 for x in c):3d}")
