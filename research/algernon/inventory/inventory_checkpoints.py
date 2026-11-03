"""Phase A provenance: enumerate recurrent checkpoints under LoopedTransformers.

Records path, size, sha256 and (when torch is importable) top-level keys and
tensor shapes. Weights are never copied; output is a JSON receipt.
Usage: python inventory_checkpoints.py <LoopedTransformers experiments dir> <out.json>
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shapes(path):
    try:
        import torch
    except ImportError:
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return {"load_error": type(exc).__name__}
    sd = obj.get("model", obj.get("state_dict", obj)) if isinstance(obj, dict) else {}
    return {
        "top_keys": sorted(obj.keys())[:12] if isinstance(obj, dict) else None,
        "tensors": {k: list(v.shape) for k, v in sd.items() if hasattr(v, "shape")},
    }


def main(root, out):
    root = Path(root)
    records = []
    for p in sorted(root.glob("*/checkpoints/*.pt")):
        records.append({
            "experiment": p.parent.parent.name,
            "file": p.name,
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "structure": shapes(p),
            "performance_recorded": False,
        })
    Path(out).write_text(json.dumps({"count": len(records), "records": records}, indent=1))
    print(len(records), "checkpoints ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
