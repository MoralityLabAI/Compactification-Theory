"""Score a fixed-slot capture's own slot readout as whole pointer steps (H-A2d diagnostic, not a registered test).

analysis.py takes the first integer of each block's readout as the state after step k. A supervised step is
`<<s->t>>`, whose first integer is s, the state after k-1 steps, so on maps where s != t that score reads as wrong.
This scores the readout after loop k against the source, the target and the whole step, per family.

    python -m research.algernon.circuit_mirror.lotus_bridge.readout_shift --run-dir <cap_dir> --roster <roster> --out x.json
"""
import argparse
import json
import re
from pathlib import Path


def score(per_case, rows, kind="diagonal", steps=range(9, 17)):
    tally = {}
    for p in per_case:
        r = rows[p["id"]]
        if r["split"] != "test":
            continue
        tr = r["trace"]
        for k, text in enumerate(p["readout_text"][kind], start=1):
            if k not in steps:
                continue
            ints = [int(x) for x in re.findall(r"\d+", text)]
            t = tally.setdefault(r["family"], {"n": 0, "source": 0, "target": 0, "step": 0})
            t["n"] += 1
            t["source"] += bool(ints) and ints[0] == tr[k - 1]
            t["target"] += bool(ints) and ints[0] == tr[k]
            t["step"] += len(ints) >= 2 and ints[0] == tr[k - 1] and ints[1] == tr[k]
    total = {key: sum(t[key] for t in tally.values()) for key in ("n", "source", "target", "step")}
    rate = lambda t: {"cases_x_steps": t["n"], "first_int_is_state_after_k_minus_1": t["source"] / t["n"],
                      "first_int_is_state_after_k": t["target"] / t["n"], "whole_step_correct": t["step"] / t["n"]}
    return {"readout": kind, "steps": [min(steps), max(steps)], "all_test": rate(total),
            "by_family": {f: rate(t) for f, t in sorted(tally.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = {c["id"]: c for c in json.loads(a.roster.read_text(encoding="utf-8"))["cases"]}
    per_case = [json.loads(f.read_text(encoding="utf-8"))["value"] for f in sorted((a.run_dir / "parts" / "full").glob("*.json"))]
    res = {"run_dir": a.run_dir.name, "cases": len(per_case),
           "diagonal_9_16": score(per_case, rows), "diagonal_1_8": score(per_case, rows, steps=range(1, 9))}
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["diagonal_9_16"]["all_test"]))


if __name__ == "__main__":
    main()
