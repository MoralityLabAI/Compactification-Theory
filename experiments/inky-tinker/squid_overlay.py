"""Fold a Tinker Inkling 2 Mini LoRA (PEFT form) into overlay safetensors for squid5090 (SQUID_OVERLAY), which has
no runtime LoRA path.

  python squid_overlay.py --peft <adapter_peft> --ckpt <model-00001-of-00001.safetensors> --out-dir <dir> [--seed 0]

Writes two files with checkpoint tensor names, dtypes and shapes, so squid's loader can take them in place of the
originals:
  attn.safetensors     attn.{wq_du,wk_dv,wv_dv,wo_ud,gate_attn,wr_du}.weight of all 24 layers: BF16 W + (alpha/r) B A,
                       rounded to nearest even (exact merge up to BF16 rounding)
  experts.safetensors  mlp.experts.{w13,w2}_weight of the NVFP4 layers (0-19): the packed E2M1 codes of W + (alpha/r) B A
                       on the checkpoint's own grid (FP8 block scales and per-expert amax unchanged), stochastically
                       rounded. Nearest rounding would keep ~1% of the delta (it is ~7% of half a grid step); stochastic
                       rounding keeps it in expectation and adds a fixed noise of ~4.4% of |W| per expert matrix.
                       w13 rows are interleaved as stored: row 2i is gate i, row 2i+1 is up i.
Layers 20-23 hold MXFP8 experts that squid re-quantizes to NVFP4 at load (nearest), which would erase a merged
delta, so their experts are left out; their attention is merged. stats.json records per-layer delta retention.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ATTN = ["wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn", "wr_du"]
GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
LUT = np.concatenate([GRID, -GRID]).astype(np.float32)
NVFP4_LAYERS = range(20)


class Ckpt:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.h.pop("__metadata__", None)
        self.base = 8 + n

    def raw(self, name: str) -> tuple[bytes, dict]:
        m = self.h[name]
        a, b = m["data_offsets"]
        self.f.seek(self.base + a)
        return self.f.read(b - a), m

    def expert_slice(self, name: str, e: int) -> np.ndarray:
        m = self.h[name]
        per = int(np.prod(m["shape"][1:]))
        self.f.seek(self.base + m["data_offsets"][0] + e * per)
        return np.frombuffer(self.f.read(per), np.uint8).reshape(m["shape"][1:])


def f8e4m3(u8: np.ndarray) -> np.ndarray:
    u = u8.astype(np.int32)
    s = np.where(u & 0x80, -1.0, 1.0)
    e, m = (u >> 3) & 0xF, u & 7
    return (s * np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))).astype(np.float32)


def to_bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class Writer:
    """Streaming safetensors writer: declare every tensor first, then write their bytes in order."""

    def __init__(self, path: Path, specs: list[tuple[str, str, list[int], int]]):
        hdr, off = {}, 0
        for name, dt, shape, nbytes in specs:
            hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nbytes]}
            off += nbytes
        h = json.dumps(hdr, separators=(",", ":")).encode()
        h += b" " * ((-len(h)) % 8)
        self.f = open(path, "wb")
        self.f.write(struct.pack("<Q", len(h)) + h)
        self.end = 8 + len(h) + off

    def write(self, b: bytes):
        self.f.write(b)

    def close(self):
        assert self.f.tell() == self.end, (self.f.tell(), self.end)
        self.f.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="", help="comma list of expert layers (default: all NVFP4 layers)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(a.peft) / "adapter_config.json").read_text())
    sc = float(cfg["lora_alpha"]) / int(cfg["r"])
    lora = load_file(str(Path(a.peft) / "adapter_model.safetensors"))
    L = lambda layer, mod, ab: lora[f"base_model.model.language_model.layers.{layer}.{mod}.lora_{ab}.weight"].astype(np.float32)
    ck = Ckpt(a.ckpt)
    stats: dict = {"scale": sc, "attn": {}, "experts": {}}

    # attention, exact up to BF16 rounding
    names = [f"model.llm.layers.{l}.attn.{m}.weight" for l in range(24) for m in ATTN]
    specs = [(n, "BF16", ck.h[n]["shape"], ck.h[n]["data_offsets"][1] - ck.h[n]["data_offsets"][0]) for n in names]
    w = Writer(out / "attn.safetensors", specs)
    for n in names:
        layer, mod = int(n.split(".")[3]), n.split(".")[5]
        b, m = ck.raw(n)
        W = (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
        D = sc * L(layer, f"attn.{mod}", "B") @ L(layer, f"attn.{mod}", "A")
        nw = to_bf16(W + D)
        eff = (nw.astype(np.uint32) << 16).view(np.float32) - W
        stats["attn"][n] = {"rel_delta": float(np.linalg.norm(D) / np.linalg.norm(W)),
                            "kept": float((eff * D).sum() / (D * D).sum())}
        w.write(nw.tobytes())
    w.close()
    print("attn overlay written", flush=True)

    # NVFP4 experts, stochastic rounding on the checkpoint grid
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(NVFP4_LAYERS)
    names = [f"model.llm.layers.{l}.mlp.experts.{t}_weight" for l in layers for t in ("w13", "w2")]
    specs = [(n, "U8", ck.h[n]["shape"], ck.h[n]["data_offsets"][1] - ck.h[n]["data_offsets"][0]) for n in names]
    w = Writer(out / "experts.safetensors", specs)
    rng = np.random.default_rng(a.seed)
    for n in names:
        t0 = time.time()
        layer, which = int(n.split(".")[3]), n.split(".")[-1].split("_")[0]
        pre = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
        amax = np.frombuffer(ck.raw(pre + "_amax")[0], np.float32)
        den = np.frombuffer(ck.raw(pre + "_outer_scale_denominator")[0], np.float32)[0]
        E = ck.h[n]["shape"][0]
        num = den2 = dd = 0.0
        for e in range(E):
            codes = ck.expert_slice(n, e)
            rows = codes.shape[0]
            q = np.stack((LUT[codes & 0xF], LUT[codes >> 4]), -1).reshape(rows, -1)
            blk = f8e4m3(ck.expert_slice(pre + "_scale", e)) * (amax[e] / den)
            step = np.repeat(blk, 16, axis=1)
            if which == "w13":
                D = np.empty_like(q)
                D[0::2] = sc * L(layer, f"mlp.experts.{e}.gate_proj", "B") @ L(layer, f"mlp.experts.{e}.gate_proj", "A")
                D[1::2] = sc * L(layer, f"mlp.experts.{e}.up_proj", "B") @ L(layer, f"mlp.experts.{e}.up_proj", "A")
            else:
                D = sc * L(layer, f"mlp.experts.{e}.down_proj", "B") @ L(layer, f"mlp.experts.{e}.down_proj", "A")
            x = q + D / np.maximum(step, 1e-30)               # target in grid units
            mag = np.minimum(np.abs(x), 6.0)
            hi = np.searchsorted(GRID, mag).clip(1, 7)
            lo = hi - 1
            p = (mag - GRID[lo]) / (GRID[hi] - GRID[lo])
            idx = np.where(rng.random(mag.shape, dtype=np.float32) < p, hi, lo)
            code = (idx + np.where((x < 0) & (idx > 0), 8, 0)).astype(np.uint8)
            packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
            w.write(packed.tobytes())
            eff = (np.sign(x) * GRID[idx] - q) * step
            num += float((eff * D).sum()); den2 += float((D * D).sum()); dd += float(((eff - D) ** 2).sum())
        stats["experts"][n] = {"kept_projection": num / den2, "noise_over_delta": (dd / den2) ** 0.5}
        print(n, json.dumps(stats["experts"][n]), f"{time.time() - t0:.0f}s", flush=True)
    w.close()
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
