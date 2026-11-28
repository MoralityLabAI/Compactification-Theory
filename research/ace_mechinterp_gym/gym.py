"""ACE-style synthetic causal-recoverability gym. Standard library; not an ACE certificate.

Run from repo root: python -m research.ace_mechinterp_gym.gym --out /tmp/ace-gym
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

from research.dynamical_closure.task_closure import (
    FiniteSystem, minimum_bits, task_loss_witness,
)

# States in order (y,b) = 00,01,10,11. Action reveal exposes hidden bit.
TASK = (0, 0, 1, 1)
ACTIONS = {"id": (0, 1, 2, 3), "reveal": (0, 3, 0, 3)}
CODES = {"full": (0, 1, 2, 3), "task": (0, 0, 1, 1),
         "constant": (0, 0, 0, 0)}
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")


def canonical_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_manifest(m: dict) -> None:
    if m.get("schema") != "ace-mechinterp-gym/v0.1" or m.get("task") != "hidden-bit-v1":
        raise ValueError("unknown protocol/task")
    if type(m.get("seed")) is not int or type(m.get("rounds_per_candidate")) is not int:
        raise ValueError("seed and rounds must be integers")
    if not 1 <= m["rounds_per_candidate"] <= 10000:
        raise ValueError("invalid round count")
    k = m.get("substitution_budget_per_candidate")
    if type(k) is not int or not 0 <= k <= m["rounds_per_candidate"]:
        raise ValueError("invalid substitution budget")
    p, q = m.get("null_violation_floor"), m.get("alternative_violation_rate")
    if type(p) not in (int, float) or type(q) not in (int, float) or not (0 < q < p < 1):
        raise ValueError("require 0 < alternative q < unsafe-null floor p < 1")
    root = Decimal(str(m.get("root_alpha")))
    if not 0 < root < 1:
        raise ValueError("root alpha out of range")
    candidates = m.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("missing candidate list")
    names, spent = set(), Decimal(0)
    for c in candidates:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"]:
            raise ValueError("invalid candidate ID")
        if c["id"] in names:
            raise ValueError("duplicate candidate ID")
        names.add(c["id"])
        if c.get("code") not in CODES or not isinstance(c.get("actions"), list):
            raise ValueError("unknown code or actions")
        if not c["actions"] or len(set(c["actions"])) != len(c["actions"]) or any(a not in ACTIONS for a in c["actions"]):
            raise ValueError("invalid action set")
        a = Decimal(str(c.get("alpha")))
        if not 0 < a < 1:
            raise ValueError("invalid child alpha")
        spent += a  # allocated once; no refunds for unsuccessful tests
    if spent > root:
        raise ValueError("child allocations exceed root risk scope")
    attacks = m.get("synthetic_attacks", [])
    if not isinstance(attacks, list):
        raise ValueError("invalid attacks")
    seen = set()
    flips = {n: 0 for n in names}
    for a in attacks:
        if (not isinstance(a, dict) or a.get("candidate") not in names or
            type(a.get("index")) is not int or not 0 <= a["index"] < m["rounds_per_candidate"] or
            a.get("kind") not in ("erase", "flip")):
            raise ValueError("invalid attack instruction")
        key = (a["candidate"], a["index"])
        if key in seen:
            raise ValueError("duplicate attack on one slot")
        seen.add(key)
        flips[a["candidate"]] += a["kind"] == "flip"
    if any(v > k for v in flips.values()):
        raise ValueError("synthetic attacker exceeds declared substitution budget")


def system_for(candidate: dict) -> FiniteSystem:
    return FiniteSystem(tuple(ACTIONS[a] for a in candidate["actions"]), TASK)


def probability_exact(candidate: dict) -> Fraction:
    """P(outputs disagree), sampling x uniform, x' uniform in C(x)-fiber,
    and intervention uniformly from candidate's precommitted actions.
    """
    code = CODES[candidate["code"]]
    actions = candidate["actions"]
    risk = Fraction(0)
    for x in range(4):
        fiber = [y for y in range(4) if code[y] == code[x]]
        for y in fiber:
            for action in actions:
                outcome = int(TASK[ACTIONS[action][x]] != TASK[ACTIONS[action][y]])
                risk += Fraction(outcome, 4 * len(fiber) * len(actions))
    return risk


def _candidate_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{name}".encode()).digest()[:8], "big")


def simulate(m: dict) -> list[dict]:
    """One fresh *synthetic* pair per cluster; replicas of a source are never extra trials."""
    validate_manifest(m)
    attacks = {(a["candidate"], a["index"]): a["kind"]
               for a in m.get("synthetic_attacks", [])}
    samples = []
    for c in m["candidates"]:
        rng = random.Random(_candidate_seed(m["seed"], c["id"]))
        code = CODES[c["code"]]
        fibers = {v: tuple(j for j, v2 in enumerate(code) if v2 == v) for v in set(code)}
        for i in range(m["rounds_per_candidate"]):
            x = rng.randrange(4)
            y = rng.choice(fibers[code[x]])
            action = rng.choice(c["actions"])
            clean = int(TASK[ACTIONS[action][x]] != TASK[ACTIONS[action][y]])
            kind = attacks.get((c["id"], i))
            reported = None if kind == "erase" else (1 - clean if kind == "flip" else clean)
            samples.append({"candidate": c["id"], "slot": i,
                            "source_cluster": f"{c['id']}/{i}", "x": x, "y": y,
                            "action": action, "clean": clean, "reported": reported})
    return samples


def robust_prefixes(reports: list[int | None], p: float, q: float, k: int) -> list[float]:
    """Exact log lower envelope for IID Bernoulli null/alternative when q<p.

    An erased bit may be 1. To minimize evidence, turn up to k reported
    zeroes into clean ones. The true clean prefix must be feasible.
    """
    if not (0 < q < p < 1) or type(k) is not int or k < 0:
        raise ValueError("invalid null, alternative or corruption budget")
    zeros = ones = missing = 0
    violation_factor = math.log(q / p)
    clear_factor = math.log((1-q)/(1-p))
    out = []
    for v in reports:
        if v is None:
            missing += 1
        elif type(v) is int and v == 0:
            zeros += 1
        elif type(v) is int and v == 1:
            ones += 1
        else:
            raise ValueError("reports must be 0, 1 or None")
        substitutions = min(k, zeros)
        out.append((ones + missing + substitutions) * violation_factor +
                   (zeros - substitutions) * clear_factor)
    return out


def first_crossing(prefixes: list[float], alpha: str) -> int | None:
    threshold = math.log(1/float(Decimal(str(alpha))))
    return next((i+1 for i, score in enumerate(prefixes) if score >= threshold - 1e-12), None)


def copied_source_counterexample(m: dict) -> dict:
    """Exactly one null Bernoulli(.25) outcome copied 32x, not 32 trials."""
    p, q = m["null_violation_floor"], m["alternative_violation_rate"]
    candidate = next((c for c in m["candidates"] if c["code"] == "task"
                      and set(c["actions"]) == {"id", "reveal"}), None)
    if candidate is None:
        return {"applicable": False, "reason": "no task/reveal candidate in fixed policy"}
    risk = probability_exact(candidate)
    unsafe_null_applies = risk >= Fraction(str(p))
    repeated = [0] * 32
    naive = first_crossing(robust_prefixes(repeated,p,q,0), candidate["alpha"])
    dedup = first_crossing(robust_prefixes([0],p,q,0), candidate["alpha"])
    clean_zero = 1 - risk
    return {"applicable": True, "copies": 32, "independent_clusters": 1,
            "single_clean_zero_probability": str(clean_zero),
            "under_unsafe_null": unsafe_null_applies,
            "naive_false_certification_probability": str(clean_zero) if naive and unsafe_null_applies else "0",
            "naive_first_crossing": naive, "cluster_aware_first_crossing": dedup}


def summarize(m: dict, samples: list[dict]) -> dict:
    validate_manifest(m)
    p, q, k = m["null_violation_floor"], m["alternative_violation_rate"], m["substitution_budget_per_candidate"]
    results = []
    for c in m["candidates"]:
        group = [s for s in samples if s["candidate"] == c["id"]]
        reports = [s["reported"] for s in group]
        logs = robust_prefixes(reports,p,q,k)
        first = first_crossing(logs, c["alpha"])
        system = system_for(c)
        witness = task_loss_witness(system, CODES[c["code"]])
        exact = probability_exact(c)
        results.append({"id": c["id"], "code": c["code"],
                        "actions": c["actions"], "code_bits_minimum": minimum_bits(system),
                        "exact_violation_rate": f"{exact.numerator}/{exact.denominator}",
                        "exact_task_witness": None if witness is None else
                            {"x": witness[0], "y": witness[1],
                             "word": [c["actions"][a] for a in witness[2]]},
                        "n_clusters": len(group),
                        "reported_violations": reports.count(1),
                        "reported_erasures": reports.count(None),
                        "alpha_spent": c["alpha"],
                        "ace_first_certification_round": first,
                        "matched_robust_bayes_first_certification_round": first,
                        "final_robust_log_e": round(logs[-1],8) if logs else None})
    return {"budget": {"root_alpha": m["root_alpha"],
                       "allocated_alpha": str(sum(Decimal(str(c["alpha"])) for c in m["candidates"]))},
            "results": results, "correlated_source_attack": copied_source_counterexample(m)}


def write_run(manifest_path: Path, out_dir: Path) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = simulate(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_data = canonical_bytes(samples)
    receipt = {"schema": "ace-mechinterp-gym-receipt/v0.1",
               "manifest_sha256": sha256(manifest_bytes),
               "trace_sha256": sha256(trace_data),
               "gym_source_sha256": sha256(Path(__file__).read_bytes()),
               "task_quotient_source_sha256": sha256(Path(__import__("research.dynamical_closure.task_closure", fromlist=["__file__"]).__file__).read_bytes()),
               "evidence_note": "synthetic pseudorandom sample pairs; unsigned local replay, no physical acquisition attestation",
               "diagnostics": summarize(manifest, samples)}
    (out_dir / "trace.json").write_bytes(trace_data)
    (out_dir / "receipt.json").write_bytes(canonical_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = write_run(args.manifest, args.out)
    print(json.dumps(receipt["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
