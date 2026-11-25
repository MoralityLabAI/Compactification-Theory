"""Minimal GGUF v3 header reader (KV + tensor infos), tolerant of unknown tensor types."""
import struct
SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
def read(path):
    f = open(path, "rb")
    def u(fmt):
        n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
    def s(): return f.read(u("<Q")).decode("utf-8", "replace")
    def val(t):
        if t in SC: return u(SC[t])
        if t == 8: return s()
        if t == 9:
            et, n = u("<I"), u("<Q"); return [val(et) for _ in range(n)]
        raise ValueError(t)
    assert f.read(4) == b"GGUF"; ver = u("<I"); nt, nk = u("<Q"), u("<Q")
    kv = {}
    for _ in range(nk):
        k = s(); t = u("<I"); kv[k] = (t, val(t))
    tens = []
    for _ in range(nt):
        name = s(); nd = u("<I"); dims = [u("<Q") for _ in range(nd)]; typ = u("<I"); off = u("<Q")
        tens.append([name, dims, typ, off])
    align = kv.get("general.alignment", (4, 32))[1]
    pos = f.tell(); data_start = (pos + align - 1) // align * align
    f.seek(0, 2); end = f.tell()
    offs = sorted(t[3] for t in tens) + [end - data_start]
    nxt = {o: offs[i + 1] for i, o in enumerate(offs[:-1])}
    for t in tens: t.append(nxt[t[3]] - t[3])  # padded byte size
    return {"version": ver, "kv": kv, "tensors": tens, "data_start": data_start, "align": align, "end": end}
