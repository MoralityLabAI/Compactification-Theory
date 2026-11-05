"""H-A2c behavioural depth curve, and the registered comparisons between arms.

`run`: for every calibration and held-out case of the frozen roster and every
d = 1..16, ask for d steps of the same map and start, and score the state at
trace[d]. A LOTUS checkpoint gets L = max(8, d) latent blocks and loops (8 is
its trained count); the stage-1 CoT model decodes freely. Answers are saved
per case, so a power-cap pause (exit 75) resumes where it stopped.

`compare`: the H-A2c decision rules from two or more depth_curve.json files.

H-A2d (`--prompt-style nocount --fixed-slots 6`): the question carries no step
count, the model gets one block of fixed slots and exactly d loops.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .gpu_run import PAUSE_EXIT, Parts, Paused, PowerWatch, log, upstream_extract_answer, write_atomic
from .pointer_loop import prompt_nocount
from .roster import prompt_for

DEPTHS = range(1, 17)
TRAINED_BLOCKS = 8


def depth_rows(roster, prompt=prompt_for):
    rows = []
    for c in roster["cases"]:
        if c["split"] == "train":
            continue
        for d in DEPTHS:
            sub = {"successors": c["successors"], "initial": c["initial"], "trace": c["trace"][:d + 1]}
            rows.append({"id": f"{c['id']}@{d}", "case": c["id"], "family": c["family"], "split": c["split"],
                         "depth": d, "question": prompt(sub).rstrip("\n"), "answer": str(c["trace"][d])})
    return rows


def case_bootstrap(per_case, n_boot=2000, seed=0):
    import numpy as np
    x = np.asarray(per_case, dtype=float)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def summarize(records):
    from collections import defaultdict
    by = defaultdict(list)
    for r in records:
        by[(r["family"], r["depth"])].append(r["correct"])
        if r["split"] == "test":
            by[("heldout", r["depth"])].append(r["correct"])
    families = sorted({r["family"] for r in records}) + ["heldout"]
    table = {f: {d: sum(by[(f, d)]) / len(by[(f, d)]) for d in DEPTHS if by[(f, d)]} for f in families}
    out = {"accuracy_by_depth": table}
    for name, keep in (("heldout", lambda r: r["split"] == "test"), ("long_cycle", lambda r: r["family"] == "long_cycle")):
        cases = defaultdict(list)
        for r in records:
            if keep(r) and r["depth"] >= 9:
                cases[r["case"]].append(r["correct"])
        per_case = [sum(v) / len(v) for _, v in sorted(cases.items())]
        out[f"{name}_9_16"] = {"accuracy": sum(per_case) / len(per_case), "ci95": case_bootstrap(per_case),
                               "cases": len(per_case)}
    return out


def run(args):
    import torch
    from .train_lotus import answers, load_trained
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    rows = depth_rows(roster, prompt_nocount if args.prompt_style == "nocount" else prompt_for)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.checkpoint.with_name("train_manifest.json").read_text(encoding="utf-8"))
    kind = "cot" if meta["phase"] == "cot" else "lotus"
    model, _, tok = load_trained(args.upstream, args.base, args.checkpoint, device=args.device)
    extract = upstream_extract_answer(args.upstream)
    parts = Parts(args.out / "parts", {"checkpoint_sha256": meta["final_sha256"], "roster": roster["roster_sha256"],
                                       "max_new_tokens": args.cot_tokens if kind == "cot" else 16,
                                       "prompt_style": args.prompt_style, "fixed_slots": args.fixed_slots})
    watch = PowerWatch(args.min_power_w, active=str(args.device).startswith("cuda"))
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.cuda_fraction, 0)
    records, t0 = [], time.perf_counter()
    try:
        for i, row in enumerate(rows):
            hit = parts.load("depth", row["id"])
            if hit is None:
                watch.check(row["id"])
                loops = row["depth"] if args.fixed_slots else max(TRAINED_BLOCKS, row["depth"])
                blocks = 1 if args.fixed_slots else loops
                res = answers(model, tok, [row], kind=kind, blocks=blocks, loops=loops, c_thought=meta["c_thought"],
                              max_new_tokens=args.cot_tokens if kind == "cot" else 16, extract=extract)[0]
                rec = {k: row[k] for k in ("id", "case", "family", "split", "depth", "answer")}
                rec |= {"predicted": res["answer"], "text": res["answer_text"], "correct": res["answer"] == row["answer"],
                        "loops": 0 if kind == "cot" else loops}
                parts.save("depth", row["id"], rec)
            else:
                rec = hit[0]
            records.append(rec)
            if (i + 1) % 128 == 0:
                log(f"depth curve {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)")
    except Paused as e:
        log(f"PAUSE: {e}; exit {PAUSE_EXIT}")
        sys.exit(PAUSE_EXIT)
    out = {"schema_version": "algernon-02c-depth-v1", "checkpoint": args.checkpoint.name, "phase": meta["phase"],
           "after_loop": meta.get("after_loop"), "checkpoint_sha256": meta["final_sha256"],
           "roster_sha256": roster["roster_sha256"],
           "rule": ("one block of fixed slots, d loops, no step count in the question" if args.fixed_slots
                    else "latent blocks = loops = max(8, d); CoT decodes freely"),
           "summary": summarize(records), "records": records}
    write_atomic(args.out / "depth_curve.json", json.dumps(out, indent=1, sort_keys=True) + "\n")
    s = out["summary"]
    log(f"held-out 9-16 {s['heldout_9_16']['accuracy']:.3f} {s['heldout_9_16']['ci95']}; "
        f"long_cycle 9-16 {s['long_cycle_9_16']['accuracy']:.3f}")


def compare(args):
    import numpy as np
    from collections import defaultdict
    curves = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in args.curves}

    def per_case(curve, keep):
        cases = defaultdict(list)
        for r in curve["records"]:
            if keep(r) and r["depth"] >= 9:
                cases[r["case"]].append(r["correct"])
        return {c: sum(v) / len(v) for c, v in cases.items()}

    heldout = lambda r: r["split"] == "test"
    result = {name: c["summary"] for name, c in curves.items()}
    if args.a and args.b:
        a, b = per_case(curves[args.a], heldout), per_case(curves[args.b], heldout)
        diff = [b[c] - a[c] for c in sorted(a)]
        result["loop_step_tying"] = {"b_minus_a_heldout_9_16": float(np.mean(diff)), "ci95": case_bootstrap(diff),
                                     "margin_required": 0.10}
        result["loop_step_tying"]["matters"] = bool(result["loop_step_tying"]["b_minus_a_heldout_9_16"] >= 0.10
                                                    and result["loop_step_tying"]["ci95"][0] > 0)
    if args.baseline is not None:
        for name, c in curves.items():
            lc = per_case(c, lambda r: r["family"] == "long_cycle")
            diff = [v - args.baseline[k] for k, v in sorted(lc.items())]
            result[name]["long_cycle_9_16_vs_best_trivial"] = {"diff": float(np.mean(diff)), "ci95": case_bootstrap(diff)}
    write_atomic(args.out, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k == "loop_step_tying"}, indent=1))


def trivial_baseline_long_cycle(roster_path: Path, metrics_path: Path):
    """Per long_cycle case, the 02B best trivial baseline's accuracy over stages 9-16 (its records)."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    name = metrics["gate"]["long_cycle_vs_best_trivial"]["baseline"]
    recs = [json.loads(line) for line in metrics_path.with_name("records.jsonl").read_text(encoding="utf-8").splitlines()]
    out = {r["case_id"]: r["correct_9_16"] / 8 for r in recs
           if r["intervention"] == name and r["family"] == "long_cycle"}
    if len(out) != 24:
        raise ValueError(f"expected 24 long_cycle records for {name}, found {len(out)}")
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--upstream", type=Path, required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--roster", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--cot-tokens", type=int, default=160)
    r.add_argument("--prompt-style", choices=("count", "nocount"), default="count")
    r.add_argument("--fixed-slots", type=int, default=0, help="H-A2d: one block of this many slots, d loops")
    r.add_argument("--min-power-w", type=float, default=50)
    r.add_argument("--cuda-fraction", type=float, default=0.70)
    c = sub.add_parser("compare")
    c.add_argument("--curves", type=Path, nargs="+", required=True)
    c.add_argument("--a", help="run-dir name of arm A")
    c.add_argument("--b", help="run-dir name of arm B")
    c.add_argument("--roster", type=Path)
    c.add_argument("--baseline-metrics", type=Path, help="02B metrics.json, for the best trivial long_cycle baseline")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        args.baseline = None
        if args.baseline_metrics:
            args.baseline_name, args.baseline = trivial_baseline_long_cycle(args.roster, args.baseline_metrics)
        compare(args)


if __name__ == "__main__":
    main()
