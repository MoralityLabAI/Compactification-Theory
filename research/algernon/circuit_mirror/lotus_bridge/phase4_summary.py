"""Phase 5 roll-up of the Phase 4 runs (H-A3 to H-A3g) from their committed receipts.

Reads each `results/algernon-03*/` directory (eval.json, train.json, manifest.json) and writes
`results/phase4/metrics.json`: per run, the settings that changed, accuracy and exact-checker trace
acceptance per arm and condition, clean damage, the registered decisions with their case-bootstrap
intervals, corrector sizes, and the receipt hashes. It reads no model and needs no GPU.

Hashes are taken over the files with CRLF folded to LF, so a Windows checkout with autocrlf and a
Linux checkout give the same values; for every run they must equal the eval.json hash in that run's
manifest, or the script stops.

    python -m research.algernon.circuit_mirror.lotus_bridge.phase4_summary          # write
    python -m research.algernon.circuit_mirror.lotus_bridge.phase4_summary --check  # compare only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = ("algernon-03", "algernon-03b", "algernon-03c", "algernon-03d", "algernon-03e", "algernon-03f", "algernon-03g")
# correct.py's argparse defaults, for runs registered before a flag existed
DEFAULTS = {"out_scale": "unit", "corrector_bias": "on", "noise_ref": "slots", "sigma_rule": "half", "loss": "local"}
GRID = {"sigma16_local": "algernon-03d", "sigma16_answer": "algernon-03g",
        "sigma1_local": "algernon-03e", "sigma1_answer": "algernon-03f"}
LOOPS = 16


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def summarise_run(name: str, results: Path = RESULTS) -> dict:
    d = results / name
    ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
    tr = json.loads((d / "train.json").read_text(encoding="utf-8"))
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    eval_sha = lf_sha256(d / "eval.json")
    recorded = man.get("repo_receipts_sha256", {}).get("eval.json")
    if recorded != eval_sha:
        raise SystemExit(f"{name}: eval.json hash {eval_sha} does not match the manifest's {recorded}")
    s, args = ev["result"]["summary"], ev["args"]
    records = ev["result"]["records"]
    return {
        "run": name,
        "experiment": ev["experiment"],
        "code_commit": man["code_commit"],
        "shared": {"checkpoint_sha256": ev["checkpoint_sha256"], "roster_sha256": ev["roster_sha256"],
                   "train_seed": args["train_seed"], "calib_seed": args["calib_seed"], "depth": args["depth"]},
        "registered_in": man["registered_in"],
        "settings": {**{k: args.get(k) or v for k, v in DEFAULTS.items()},
                     "sigma_primary": ev["sigma_star"], "kappa": ev.get("kappa"),
                     "noise_loops": args["noise_loops"], "noise_seeds": args["seeds"],
                     "train_steps": args["steps"], "train_batch": args["batch"]},
        "accuracy": s["accuracy"],
        "trace_valid": s["trace_valid"],
        "accuracy_by_noise_loop_primary": s["by_j"],
        "accuracy_by_family_primary": s["by_family"],
        "clean_damage_below_0.97": s["clean_damage"],
        "decisions": s["decisions"],
        "records": {"count": len(records),
                    "per_arm_condition": _counts(records)},
        "compute": {"f_forward_passes_per_run": LOOPS + 1,
                    "corrector_parameters": {a: tr["result"][a]["parameters"] for a in ("C", "I", "N")}},
        "receipts_lf_sha256": {f: lf_sha256(d / f) for f in ("eval.json", "train.json", "manifest.json")},
    }


def _counts(records) -> dict:
    out: dict = {}
    for r in records:
        out.setdefault(r["arm"], {}).setdefault(r["condition"], 0)
        out[r["arm"]][r["condition"]] += 1
    return out


def build(results: Path = RESULTS) -> dict:
    runs = [summarise_run(n, results) for n in RUNS]
    by = {r["run"]: r for r in runs}
    shared = {json.dumps(r.pop("shared"), sort_keys=True) for r in runs}
    if len(shared) != 1:
        raise SystemExit(f"the runs differ in checkpoint, roster, seeds or depth: {sorted(shared)}")
    grid = {cell: {"run": run, "primary_accuracy": {a: by[run]["accuracy"][a]["primary"] for a in "PCIN"},
                   "met": [k.split("_")[0] for k, v in by[run]["decisions"].items() if v["met"]]}
            for cell, run in GRID.items()}
    return {"schema_version": "algernon-phase4-summary-v1",
            "generated_by": "research/algernon/circuit_mirror/lotus_bridge/phase4_summary.py",
            "shared_by_every_run": json.loads(shared.pop()),
            "model": json.loads((results / "algernon-03f" / "manifest.json").read_text(encoding="utf-8"))["model"],
            "grid_noise_by_loss": grid,
            "runs": runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--check", action="store_true", help="compare with the committed metrics.json and write nothing")
    args = ap.parse_args(argv)
    out = args.results / "phase4" / "metrics.json"
    text = json.dumps(build(args.results), indent=1, sort_keys=True) + "\n"
    if args.check:
        same = out.read_bytes().replace(b"\r\n", b"\n").decode("utf-8") == text
        print("metrics.json matches the receipts" if same else "metrics.json differs from the receipts")
        return 0 if same else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
