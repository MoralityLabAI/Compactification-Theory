"""H-A2d: the loop count as the only depth signal, over one fixed set of latent slots.

`prompt_nocount` is `roster.prompt_for` without the step count. `convert` rewrites
the 02C rows (same maps, starts and traces) to it. `patched_lotus_module` loads a
copy of upstream `scripts/lotus.py` with one line changed: per-loop supervision
indexes the same slots on every loop (`block_start = 0`) when the model's
`fixed_slot_supervision` flag is set. The file on disk is not touched, inference
is unchanged, and the patched source hash goes into the manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

from ..spec import digest

PATCH_FROM = "                block_start = loop_idx * c_thought\n"
PATCH_TO = ("                block_start = 0 if getattr(self, \"fixed_slot_supervision\", False) "
            "else loop_idx * c_thought\n")


def prompt_nocount(row):
    """No target trace, answer or step count appears in this model input."""
    mapping = ", ".join(f"{i}->{j}" for i, j in enumerate(row["successors"]))
    return ("A machine has numbered states and one deterministic successor per state. "
            f"The successor mapping is {mapping}. Starting at state {row['initial']}, "
            "follow the successor once per loop. Which state is reached?\n")


def patched_source(upstream: Path):
    src = (upstream / "scripts" / "lotus.py").read_text(encoding="utf-8")
    if src.count(PATCH_FROM) != 1:
        raise ValueError("upstream lotus.py no longer has exactly one per-loop block_start line")
    return src.replace(PATCH_FROM, PATCH_TO)


def patched_lotus_module(upstream: Path):
    """(module, sha256 of the patched source)."""
    src = patched_source(upstream)
    name = "algernon_lotus_fixed_slots"
    module = types.ModuleType(name)
    module.__file__ = str(upstream / "scripts" / "lotus.py")
    sys.modules[name] = module
    exec(compile(src, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(src.encode("utf-8")).hexdigest()


def convert(rows):
    return [{**r, "question": prompt_nocount(r).rstrip("\n")} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", type=Path, required=True, help="02C data directory")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.from_dir / "pointer_data.json").read_text(encoding="utf-8"))
    out = {}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pointer_train.json", "pointer_val.json"):
        rows = convert(json.loads((a.from_dir / name).read_text(encoding="utf-8")))
        (a.out_dir / name).write_text(json.dumps(rows, sort_keys=True) + "\n", encoding="utf-8")
        out[name] = digest(rows)
    new = {"schema_version": "algernon-02d-pointer-loop-v1", "parameters": {**meta["parameters"], "prompt": "nocount"},
           "from_02c": {"train_sha256": meta["train_sha256"], "val_sha256": meta["val_sha256"]},
           "train_sha256": out["pointer_train.json"], "val_sha256": out["pointer_val.json"]}
    (a.out_dir / "pointer_data.json").write_text(json.dumps(new, sort_keys=True) + "\n", encoding="utf-8")
    print(f"train {new['train_sha256'][:12]} val {new['val_sha256'][:12]}")


if __name__ == "__main__":
    main()
