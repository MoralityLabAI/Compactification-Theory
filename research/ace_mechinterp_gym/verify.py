"""Local deterministic receipt replay, not a secure attestation system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gym import ACTIONS, CODES, TASK, sha256, summarize, validate_manifest


def verify(manifest_path: Path, out_dir: Path) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    m = json.loads(manifest_bytes)
    validate_manifest(m)
    trace_bytes = (out_dir / "trace.json").read_bytes()
    receipt = json.loads((out_dir / "receipt.json").read_bytes())
    trace = json.loads(trace_bytes)
    if receipt.get("schema") != "ace-mechinterp-gym-receipt/v0.1":
        raise ValueError("incorrect receipt protocol")
    if receipt.get("manifest_sha256") != sha256(manifest_bytes):
        raise ValueError("manifest digest mismatch")
    if receipt.get("trace_sha256") != sha256(trace_bytes):
        raise ValueError("trace digest mismatch")
    # Source hash ensures code under audit is the code named in the receipt.
    from . import gym
    if receipt.get("gym_source_sha256") != sha256(Path(gym.__file__).read_bytes()):
        raise ValueError("evaluator source mismatch")
    from research.dynamical_closure import task_closure
    if receipt.get("task_quotient_source_sha256") != sha256(Path(task_closure.__file__).read_bytes()):
        raise ValueError("quotient source mismatch")
    attacks = {(a["candidate"], a["index"]): a["kind"] for a in m.get("synthetic_attacks", [])}
    expected = [(c["id"], i) for c in m["candidates"] for i in range(m["rounds_per_candidate"])]
    if len(trace) != len(expected):
        raise ValueError("trace length differs from reserved slots")
    for row, (candidate_id,i) in zip(trace, expected):
        if not isinstance(row, dict) or row.get("candidate") != candidate_id or row.get("slot") != i:
            raise ValueError("candidate/order/slot mismatch")
        if row.get("source_cluster") != f"{candidate_id}/{i}":
            raise ValueError("missing or repeated cluster identity")
        candidate = next(c for c in m["candidates"] if c["id"] == candidate_id)
        x,y,a = row.get("x"),row.get("y"),row.get("action")
        if type(x) is not int or type(y) is not int or not (0<=x<4 and 0<=y<4):
            raise ValueError("invalid sampled states")
        if CODES[candidate["code"]][x] != CODES[candidate["code"]][y]:
            raise ValueError("samples from different representation fibers")
        if a not in candidate["actions"]:
            raise ValueError("action outside pinned grammar")
        clean = int(TASK[ACTIONS[a][x]] != TASK[ACTIONS[a][y]])
        if type(row.get("clean")) is not int or row["clean"] != clean:
            raise ValueError("untrue clean measurement")
        kind = attacks.get((candidate_id,i))
        expected_report = None if kind == "erase" else (1-clean if kind == "flip" else clean)
        if (type(row.get("reported")) is not (type(None) if expected_report is None else int)
                or row.get("reported", "absent") != expected_report):
            raise ValueError("reported outcome differs from committed synthetic attack schedule")
    replay = summarize(m,trace)
    if receipt.get("diagnostics") != replay:
        raise ValueError("replayed diagnostics differ")
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.manifest,args.out)
    print(f"PASS replay: {len(report['results'])} candidates; allocated alpha {report['budget']['allocated_alpha']}")


if __name__ == "__main__":
    main()
