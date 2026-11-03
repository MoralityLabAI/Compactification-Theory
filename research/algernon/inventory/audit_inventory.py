"""Audit LoopedTransformers checkpoint inventory; deduplicate and require source-backed receipts.

This is metadata validation. No checkpoint is loaded or called a graft candidate.
From the repo root:
python -m research.algernon.inventory.audit_inventory \
    --inventory research/algernon/inventory/checkpoint_inventory.json \
    --out runs/algernon/checkpoint_audit.json

Optional: --receipts /local/receipts.json --logs-root /local/LoopedTransformers/experiments
A performance receipt qualifies only when its referenced log bytes hash-match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def check_sha(value, length=64):
    return isinstance(value, str) and len(value) == length and all(
        c in "0123456789abcdef" for c in value
    )


def tensor_parameters(tensors):
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("missing tensor shapes")
    total = 0
    for name, shape in tensors.items():
        if not name or not isinstance(shape, list) or not shape or any(
            type(d) is not int or d <= 0 for d in shape
        ):
            raise ValueError(f"bad shape: {name}")
        prod = 1
        for d in shape:
            prod *= d
        total += prod
    return total


def verified_receipt(receipt, checkpoint_sha, log_root):
    """Return (valid, reason); require matching local log bytes, not log claims alone."""
    if receipt is None:
        return False, "performance receipt missing"
    if receipt.get("checkpoint_sha256") != checkpoint_sha:
        return False, "checkpoint hash differs"
    required = ("source_log_path", "source_log_sha256", "code_revision",
                "dataset_id", "dataset_sha256", "split_sha256",
                "optimizer_recipe", "recurrence_schedule", "evaluation")
    if any(not receipt.get(k) for k in required):
        return False, "performance receipt incomplete"
    if not all(check_sha(receipt[k]) for k in
               ("source_log_sha256", "dataset_sha256", "split_sha256")):
        return False, "invalid 64-digit source hash"
    if not check_sha(receipt["code_revision"], 40):
        return False, "invalid full code commit SHA"
    if not isinstance(receipt["optimizer_recipe"], dict) or not receipt["optimizer_recipe"].get("name"):
        return False, "optimizer recipe absent"
    if not isinstance(receipt["recurrence_schedule"], dict) or not receipt["recurrence_schedule"].get("steps"):
        return False, "recurrence schedule absent"
    ev = receipt["evaluation"]
    if not isinstance(ev, dict) or ev.get("split") != "held_out" or type(ev.get("n")) is not int or ev["n"] <= 0:
        return False, "held-out evaluation not documented"
    if not isinstance(ev.get("metric"), str) or not ev["metric"]:
        return False, "evaluation metric absent"
    if type(ev.get("value")) not in (float, int):
        return False, "evaluation score absent"
    if log_root is None:
        return False, "local log root needed for byte-level verification"
    root = Path(log_root).resolve()
    relative = Path(receipt["source_log_path"])
    if relative.is_absolute() or ".." in relative.parts:
        return False, "log path escapes source root"
    log = (root / relative).resolve()
    if not log.is_relative_to(root) or not log.is_file():
        return False, "source log inaccessible"
    h = hashlib.sha256()
    with log.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != receipt["source_log_sha256"]:
        return False, "source log hash mismatch"
    return True, "source-verified candidate; model behavior still untested here"


def audit_inventory(inventory, receipts=None, log_root=None):
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != inventory.get("count"):
        raise ValueError("inventory count differs from rows")
    by_sha = defaultdict(list)
    seen_paths = set()
    for r in records:
        if not isinstance(r, dict) or not check_sha(r.get("sha256")):
            raise ValueError("invalid SHA in checkpoint inventory")
        if type(r.get("bytes")) is not int or r["bytes"] <= 0:
            raise ValueError("invalid checkpoint byte count")
        rel = f"{r.get('experiment')}/checkpoints/{r.get('file')}"
        if "/" in str(r.get("experiment", "")) or "/" in str(r.get("file", "")):
            raise ValueError("invalid relative checkpoint path")
        if rel in seen_paths:
            raise ValueError("duplicate inventory path")
        seen_paths.add(rel)
        structure = r.get("structure") or {}
        shapes = structure.get("tensors")
        param_count = tensor_parameters(shapes)
        by_sha[r["sha256"]].append((rel, r["bytes"], shapes, param_count))
    receipt_map = {}
    for item in receipts or []:
        if not isinstance(item, dict) or not check_sha(item.get("checkpoint_sha256")):
            raise ValueError("bad receipt checkpoint hash")
        key = item["checkpoint_sha256"]
        if key in receipt_map:
            raise ValueError("multiple receipts for one unique checkpoint")
        receipt_map[key] = item

    grouped = []
    for digest, aliases in sorted(by_sha.items()):
        if len({(n, json.dumps(s, sort_keys=True), p) for _, n, s, p in aliases}) != 1:
            raise ValueError("same SHA reported with inconsistent shapes/bytes")
        valid, reason = verified_receipt(receipt_map.get(digest), digest, log_root)
        grouped.append({
            "checkpoint_sha256": digest,
            "aliases": [a[0] for a in aliases],
            "checkpoint_bytes": aliases[0][1],
            "parameter_count": aliases[0][3],
            "tensor_count": len(aliases[0][2]),
            "performance_receipt_verified": valid,
            "evidence_status": reason,
        })
    return {
        "schema_version": "0.1.0",
        "execution_class": "repository_inventory_metadata_only",
        "inventory_files": len(records),
        "distinct_checkpoints": len(grouped),
        "duplicate_aliases": len(records) - len(grouped),
        "source_verified_graft_candidates": sum(x["performance_receipt_verified"] for x in grouped),
        "checkpoint_groups": grouped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--receipts", type=Path)
    ap.add_argument("--logs-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    receipts = json.loads(args.receipts.read_text(encoding="utf-8")) if args.receipts else None
    if isinstance(receipts, dict):
        receipts = receipts.get("receipts")
    out = audit_inventory(inventory, receipts, args.logs_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{out['inventory_files']} files, {out['distinct_checkpoints']} unique, "
          f"{out['source_verified_graft_candidates']} evidence-qualified; {args.out}")


if __name__ == "__main__":
    main()
