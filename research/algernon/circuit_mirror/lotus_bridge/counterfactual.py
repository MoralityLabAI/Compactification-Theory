"""Relabelled copies of the held-out roster cases, for an equivariance control.

Each test case (successors f, start s) gets a fixed derangement sigma of the
state labels and becomes f' = sigma . f . sigma^-1 with start sigma(s), so its
trace is sigma(trace). The graph shape, family and depth are unchanged; only
the names in the prompt move. A readout that tracks the computation described
by the prompt must follow sigma; one that has learned label or family priors
will not. No model dependencies; the output is a deterministic function of the
frozen roster and the seed. A relabelled case never equals a train or
calibration instance. It may equal another held-out instance (the sink family
has only n*n instances in total); such duplicates are recorded in
`duplicates`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..spec import digest
from .roster import validate_roster

SCHEMA = "algernon-lotus-pointer-counterfactual-v1"


def derangement(n: int, rng: random.Random):
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(i != v for i, v in enumerate(p)):
            return p


def relabel(row, sigma):
    n = len(row["successors"])
    inverse = [0] * n
    for i, v in enumerate(sigma):
        inverse[v] = i
    succ = [sigma[row["successors"][inverse[j]]] for j in range(n)]
    return succ, sigma[row["initial"]], [sigma[s] for s in row["trace"]]


def build_counterfactual(roster, *, seed: int = 20261104):
    validate_roster(roster)
    fitted = {(tuple(r["successors"]), r["initial"]) for r in roster["cases"]
              if r["split"] != "test"}
    heldout = {(tuple(r["successors"]), r["initial"]): r["id"] for r in roster["cases"]
               if r["split"] == "test"}
    cases = []
    for row in roster["cases"]:
        if row["split"] != "test":
            continue
        rng = random.Random(f"counterfactual:{seed}:{row['id']}")
        for _ in range(1000):
            sigma = derangement(len(row["successors"]), rng)
            succ, start, trace = relabel(row, sigma)
            if (tuple(succ), start) not in fitted:
                break
        else:
            raise ValueError(f"no relabelling of {row['id']} avoids train/calibration")
        key = (tuple(succ), start)
        dup = [heldout[key]] if key in heldout else []
        dup += [c["id"] for c in cases if (tuple(c["successors"]), c["initial"]) == key]
        cases.append({"id": "cf/" + row["id"], "parent_id": row["id"], "split": "counterfactual",
                      "family": row["family"], "sigma": sigma, "successors": succ,
                      "initial": start, "trace": trace, "duplicates": dup})
    params = {"seed": seed, "parent_roster_sha256": roster["roster_sha256"]}
    return {"schema_version": SCHEMA, "parameters": params,
            "counterfactual_sha256": digest({"parameters": params, "cases": cases}),
            "cases": cases}


def validate_counterfactual(cf, roster):
    if cf.get("schema_version") != SCHEMA:
        raise ValueError("unsupported counterfactual schema")
    if cf["parameters"]["parent_roster_sha256"] != roster["roster_sha256"]:
        raise ValueError("counterfactual built from a different roster")
    if cf["counterfactual_sha256"] != digest({"parameters": cf["parameters"], "cases": cf["cases"]}):
        raise ValueError("counterfactual hash mismatch")
    parents = {r["id"]: r for r in roster["cases"]}
    fitted = {(tuple(r["successors"]), r["initial"]) for r in roster["cases"]
              if r["split"] != "test"}
    for row in cf["cases"]:
        parent = parents[row["parent_id"]]
        if parent["split"] != "test" or parent["family"] != row["family"]:
            raise ValueError("counterfactual parent must be a held-out case of the same family")
        sigma = row["sigma"]
        if sorted(sigma) != list(range(len(sigma))) or any(i == v for i, v in enumerate(sigma)):
            raise ValueError("sigma must be a derangement")
        if relabel(parent, sigma) != (row["successors"], row["initial"], row["trace"]):
            raise ValueError("counterfactual case is not the sigma-relabelled parent")
        state, trace = row["initial"], [row["initial"]]
        for _ in range(len(row["trace"]) - 1):
            state = row["successors"][state]
            trace.append(state)
        if trace != row["trace"]:
            raise ValueError("invalid counterfactual trace")
        if (tuple(row["successors"]), row["initial"]) in fitted:
            raise ValueError("counterfactual instance equals a train or calibration instance")
    return len(cf["cases"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    roster = json.loads(a.roster.read_text(encoding="utf-8"))
    obj = build_counterfactual(roster, seed=a.seed)
    validate_counterfactual(obj, roster)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"frozen {len(obj['cases'])} relabelled cases; {obj['counterfactual_sha256']}; {a.out}")


if __name__ == "__main__":
    main()
