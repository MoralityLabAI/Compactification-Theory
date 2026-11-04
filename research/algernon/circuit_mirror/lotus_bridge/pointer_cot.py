"""H-A2c training data: pointer traces as LOTUS CoT rows ({question, steps, answer}).

`random_function` cases with depth 1..8 (cycling through the depths), every
(map, start) pair distinct and none taken from the frozen roster or its
counterfactual. The question is `roster.prompt_for` without its final newline:
upstream `dataset.get_dataset` appends one, so the tokenized question equals
what the 02B capture feeds the model. Each step is `<<s->t>>`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..spec import case, digest
from .roster import prompt_for

SEED = 20261105
NODES = 6
MAX_DEPTH = 8


def step_text(s: int, t: int) -> str:
    return f"<<{s}->{t}>>"


def row_for(successors, initial, depth):
    trace = [initial]
    for _ in range(depth):
        trace.append(successors[trace[-1]])
    base = {"successors": list(successors), "initial": initial, "trace": trace}
    return {"question": prompt_for(base).rstrip("\n"),
            "steps": [step_text(a, b) for a, b in zip(trace, trace[1:])],
            "answer": str(trace[-1]), "depth": depth, **base}


def build(*, roster, counterfactual, n_train=20000, n_val=512, seed=SEED, n=NODES, max_depth=MAX_DEPTH):
    banned = {(tuple(c["successors"]), c["initial"]) for c in roster["cases"] + counterfactual["cases"]}
    seen, rows, idx = set(), [], 0
    while len(rows) < n_train + n_val:
        depth = 1 + idx % max_depth
        task = case("random_function", seed, idx, n=n, steps=depth)
        idx += 1
        key = (task.successors, task.initial)
        if key in banned or key in seen:
            continue
        seen.add(key)
        rows.append(row_for(task.successors, task.initial, depth))
    train, val = rows[:n_train], rows[n_train:]
    params = {"seed": seed, "n": n, "max_depth": max_depth, "n_train": n_train, "n_val": n_val,
              "family": "random_function", "roster_sha256": roster["roster_sha256"],
              "counterfactual_sha256": counterfactual["counterfactual_sha256"], "generator_calls": idx}
    return train, val, {"schema_version": "algernon-02c-pointer-cot-v1", "parameters": params,
                        "train_sha256": digest(train), "val_sha256": digest(val)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--counterfactual", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=512)
    a = ap.parse_args()
    roster = json.loads(a.roster.read_text(encoding="utf-8"))
    cf = json.loads(a.counterfactual.read_text(encoding="utf-8"))
    train, val, meta = build(roster=roster, counterfactual=cf, n_train=a.n_train, n_val=a.n_val)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("pointer_train.json", train), ("pointer_val.json", val), ("pointer_data.json", meta)):
        (a.out_dir / name).write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(train)} train, {len(val)} val; train {meta['train_sha256'][:12]} val {meta['val_sha256'][:12]}")


if __name__ == "__main__":
    main()
