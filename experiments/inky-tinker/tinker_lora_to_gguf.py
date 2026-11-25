"""Convert a Tinker Inkling 2 Mini LoRA (PEFT form, from tinker_cookbook.weights.build_lora_adapter) to a llama.cpp
GGUF LoRA for the inkling2_loop GGUFs (e.g. unsloth Mainline_Support Inkling-2-Mini-UD-Q4_K_M).

  python tinker_lora_to_gguf.py --peft <adapter_peft dir> --out lora.gguf [--gguf-py <llama.cpp>/gguf-py]

llama.cpp's convert_lora_to_gguf.py cannot map this adapter: the PEFT keys keep Tinker's module names
(language_model.layers.N.attn.wq_du, mlp.experts.E.gate_proj, ...) and the GGUF fuses each layer's 128 experts
into 3D tensors, with gate and up fused into one ffn_gate_up_exps. Mapping used here:

  attn.wq_du -> blk.N.attn_q      attn.wk_dv -> blk.N.attn_k      attn.wv_dv -> blk.N.attn_v
  attn.wo_ud -> blk.N.attn_output attn.gate_attn -> blk.N.attn_gate attn.wr_du -> blk.N.attn_rel
  experts.E.down_proj -> blk.N.ffn_down_exps: A [E, r, ff], B [E, d, r]
  experts.E.{gate,up}_proj -> blk.N.ffn_gate_up_exps: rows [gate | up] (the base converter de-interleaves w13 into
      gate then up), exact as a rank-2r LoRA: A = [A_gate; A_up], B = blockdiag(B_gate, B_up)

llama.cpp scales every tensor's delta by alpha / rank with rank read from that tensor's B, so the fused gate_up
B is multiplied by 2 (rank 2r instead of r) to keep PEFT's alpha / r. Tensors are written F16 (A and B are
small) in ggml order (numpy shapes reversed): 2D A [r, in], B [out, r]; 3D A [E, r, in], B [E, out, r].
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ATTN = {"wq_du": "attn_q", "wk_dv": "attn_k", "wv_dv": "attn_v", "wo_ud": "attn_output",
        "gate_attn": "attn_gate", "wr_du": "attn_rel"}
KEY = re.compile(r"language_model\.layers\.(\d+)\.(attn\.(\w+)|mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj))\.lora_([AB])\.weight$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=r"C:\Users\patri\tools\llama.cpp-b12686\gguf-py")
    ap.add_argument("--arch", default="inkling2_loop")
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf

    peft = Path(a.peft)
    cfg = json.loads((peft / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    st = load_file(str(peft / "adapter_model.safetensors"))
    attn: dict[tuple[int, str], dict[str, np.ndarray]] = defaultdict(dict)
    exp: dict[tuple[int, str], dict[int, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    unmapped = []
    for k, v in st.items():
        m = KEY.search(k)
        if not m:
            unmapped.append(k)
            continue
        layer, ab = int(m.group(1)), m.group(6)
        v = v.astype(np.float32)
        if m.group(3):
            if m.group(3) not in ATTN:
                unmapped.append(k)
                continue
            attn[(layer, ATTN[m.group(3)])][ab] = v
        else:
            exp[(layer, m.group(5))][int(m.group(4))][ab] = v
    if unmapped:
        raise SystemExit(f"{len(unmapped)} unmapped adapter tensors, e.g. {unmapped[:3]}")

    w = gguf.GGUFWriter(a.out, arch=a.arch)
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", alpha)
    n = 0

    def put(name: str, A: np.ndarray, B: np.ndarray):
        nonlocal n
        assert A.shape[-2] == B.shape[-1], (name, A.shape, B.shape)
        w.add_tensor(name + ".lora_a", np.ascontiguousarray(A.astype(np.float16)))
        w.add_tensor(name + ".lora_b", np.ascontiguousarray(B.astype(np.float16)))
        n += 2

    for (layer, tname), d in sorted(attn.items()):
        put(f"blk.{layer}.{tname}.weight", d["A"], d["B"])
    layers = sorted({l for l, _ in exp})
    for layer in layers:
        down, gate, up = exp[(layer, "down_proj")], exp[(layer, "gate_proj")], exp[(layer, "up_proj")]
        E = len(down)
        assert sorted(down) == sorted(gate) == sorted(up) == list(range(E)), layer
        A_d = np.stack([down[e]["A"] for e in range(E)])                      # [E, r, ff]
        B_d = np.stack([down[e]["B"] for e in range(E)])                      # [E, d, r]
        put(f"blk.{layer}.ffn_down_exps.weight", A_d, B_d)
        A_gu = np.stack([np.concatenate([gate[e]["A"], up[e]["A"]], 0) for e in range(E)])   # [E, 2r, d]
        ff = gate[0]["B"].shape[0]
        B_gu = np.zeros((E, 2 * ff, 2 * r), np.float32)                        # [E, 2ff, 2r] block diagonal
        for e in range(E):
            B_gu[e, :ff, :r] = gate[e]["B"]
            B_gu[e, ff:, r:] = up[e]["B"]
        put(f"blk.{layer}.ffn_gate_up_exps.weight", A_gu, B_gu * 2.0)          # alpha/(2r) * 2 = alpha/r
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(json.dumps({"out": a.out, "tensors": n, "attn_pairs": len(attn), "expert_layers": len(layers),
                      "rank": r, "alpha": alpha, "bytes": Path(a.out).stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
