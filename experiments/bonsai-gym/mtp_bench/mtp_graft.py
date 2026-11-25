"""python mtp_graft.py <Bonsai-2 PTQ1_0.gguf> <Bonsai-2.5 PTQ1_0-mtp.gguf> <out.gguf>

Graft Bonsai 2.5's MTP (nextn) head onto Bonsai 2 27B PTQ1_0: Bonsai 2's KV and 851 tensors byte for byte, plus the
15 blk.64 tensors from the 2.5 -mtp GGUF, with qwen35.block_count 64 -> 65 and qwen35.nextn_predict_layers = 1."""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufhdr import read
base_p, mtp_p, out_p = sys.argv[1:4]
B, M = read(base_p), read(mtp_p)
SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
def enc_str(s): b = s.encode(); return struct.pack("<Q", len(b)) + b
# re-read KV with array element types preserved
def read_kv_raw(path):
    f = open(path, "rb"); f.read(4); struct.unpack("<I", f.read(4)); nt, nk = struct.unpack("<QQ", f.read(16))
    def u(fmt): n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
    def s(): return f.read(u("<Q")).decode()
    def skip(t):
        if t in SC: f.read(struct.calcsize(SC[t]))
        elif t == 8: f.read(u("<Q"))
        elif t == 9:
            et, n = u("<I"), u("<Q")
            if et in SC: f.read(struct.calcsize(SC[et]) * n)
            else: [skip(et) for _ in range(n)]
    out = []
    for _ in range(nk):
        st = f.tell(); k = s(); t = u("<I"); skip(t); en = f.tell()
        f.seek(st); out.append((k, f.read(en - st)))
    return out
kvs = read_kv_raw(base_p)
newkv = []
for k, raw in kvs:
    if k == "qwen35.block_count": raw = enc_str(k) + struct.pack("<I", 4) + struct.pack("<I", 65)
    newkv.append(raw)
newkv.append(enc_str("qwen35.nextn_predict_layers") + struct.pack("<I", 4) + struct.pack("<I", 1))
newkv.append(enc_str("general.description") + struct.pack("<I", 8) + enc_str("Ternary Bonsai 2 27B PTQ1_0 + Ternary Bonsai 2.5 27B MTP head (blk.64) grafted"))
tens = [(t[0], t[1], t[2], base_p, B["data_start"] + t[3], t[4]) for t in B["tensors"]]
tens += [(t[0], t[1], t[2], mtp_p, M["data_start"] + t[3], t[4]) for t in M["tensors"] if t[0].startswith("blk.64.")]
assert len(tens) == 866, len(tens)
align = 32
hdr = b"GGUF" + struct.pack("<IQQ", 3, len(tens), len(newkv)) + b"".join(newkv)
off, infos = 0, []
for name, dims, typ, src, so, size in tens:
    infos.append(enc_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off))
    off += (size + align - 1) // align * align
hdr += b"".join(infos)
hdr += b"\0" * ((-len(hdr)) % align)
fs = {base_p: open(base_p, "rb"), mtp_p: open(mtp_p, "rb")}
with open(out_p, "wb") as o:
    o.write(hdr)
    for name, dims, typ, src, so, size in tens:
        f = fs[src]; f.seek(so); left = size
        while left: chunk = f.read(min(left, 1 << 26)); o.write(chunk); left -= len(chunk)
        o.write(b"\0" * ((-size) % align))
print("wrote", out_p, os.path.getsize(out_p))
