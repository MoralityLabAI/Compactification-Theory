"""Save/load trained adapters (safetensors) and export them as a llama.cpp GGUF LoRA.

GGUF LoRA layout as llama.cpp's convert_lora_to_gguf.py writes it and
llama-adapter.cpp reads it: general.type "adapter", adapter.type "lora",
adapter.lora.alpha; tensors "<base>.lora_a" with ggml dims [in, r] and
"<base>.lora_b" with dims [r, out], F32 here. The fork applies them in
build_lora_mm to the unrotated input, which is the basis they were trained in.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .model import PackedQwen35, load_lora_state, lora_state


def save_adapter(model: PackedQwen35, path: str | Path, meta: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = lora_state(model)
    save_file(state, str(path), metadata={"bonsai_gym": json.dumps(meta)})
    return {"path": str(path), "bytes": path.stat().st_size, "tensors": len(state),
            "params": int(sum(t.numel() for t in state.values()))}


def load_adapter(model: PackedQwen35, path: str | Path) -> int:
    return load_lora_state(model, load_file(str(path)))


def _kv_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf_lora(state: dict[str, torch.Tensor], path: str | Path, alpha: float, arch: str = "qwen35") -> dict:
    """Minimal GGUF v3 writer for an F32 LoRA (stdlib + numpy)."""
    path = Path(path)
    kvs = [("general.architecture", 8, arch), ("general.type", 8, "adapter"),
           ("adapter.type", 8, "lora"), ("adapter.lora.alpha", 6, float(alpha))]
    tensors = []
    for name in sorted(state):
        base, kind = name.rsplit(".", 1)          # "blk.0.attn_qkv.weight", "lora_a"
        arr = state[name].float().numpy()
        tensors.append((f"{base}.{kind}", arr))
    align = 32
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for k, t, v in kvs:
        head += _kv_string(k) + struct.pack("<I", t)
        head += _kv_string(v) if t == 8 else struct.pack("<f", v)
    offset, offsets = 0, []
    for name, arr in tensors:
        dims = tuple(reversed(arr.shape))         # ggml order: contiguous dim first
        head += _kv_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", 0, offset)     # type F32
        offsets.append(offset)
        offset += -(-arr.nbytes // align) * align
    head += b"\0" * (-len(head) % align)
    with open(path, "wb") as f:
        f.write(head)
        for name, arr in tensors:
            data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            f.write(data + b"\0" * (-len(data) % align))
    return {"path": str(path), "bytes": path.stat().st_size, "tensors": len(tensors), "alpha": alpha}
