"""Score one arm's replies on every family's eval split, and build rejection-sampling (RFT) training rows.

  python score_all.py score --work C:/Users/patri/Documents/Codex/onebit-suite --arm base
      reads  <work>/gen/<arm>/<family>.jsonl   (generate.py output on data/<family>/eval.jsonl)
      writes <work>/scores/<arm>/<family>.jsonl ({"id", "score", ...}) and prints per-family means

  python score_all.py rft --work ... --family math --val 0      (logic: --val 24)
      reads  data/<family>/rft_pool.jsonl and gen/rft/<family>.jsonl, keeps per prompt the shortest reply that
      scores 1 and finished with finish_reason "stop", writes data/<family>/rft_train.jsonl (+ rft_val.jsonl)

Scorers: math/logic by math_logic_score.score (the i3 envs' verifiers), srb by srb_score.score (the bench's own
validation and Bayes-optimal action), bdb by bitcoin-defi-bench scripts/score_sft_replies.ts (one-action episode
against the teacher value). Run with the onebit-suite venv (math-verify, sympy).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
NODE = r"C:\Users\patri\AppData\Local\Programs\nodejs\node.exe"
BENCH = Path(r"C:\Users\patri\OneDrive\Documents\GitHub\bitcoin-defi-bench")
FAMILIES = ["math", "logic", "srb", "bdb"]


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")


def score_rows(family: str, rows: list[dict], replies: dict[str, str], tmp: Path) -> list[dict]:
    if family == "bdb":
        rp, op = tmp / "bdb_replies.jsonl", tmp / "bdb_scored.jsonl"
        write(rp, [{"id": i, "reply": t} for i, t in replies.items()])
        rows_p = tmp / "bdb_rows.jsonl"
        write(rows_p, rows)
        subprocess.run([NODE, "--experimental-strip-types", "scripts/score_sft_replies.ts", "--rows", str(rows_p),
                        "--replies", str(rp), "--out", str(op)], cwd=BENCH, check=True, capture_output=True)
        return read(op)
    if family == "srb":
        from srb_score import score
    else:
        from math_logic_score import score
    out = []
    for r in rows:
        if r["id"] not in replies:
            out.append({"id": r["id"], "score": 0.0, "why": "no reply"})
            continue
        res = score(r, replies[r["id"]])
        out.append({"id": r["id"], **{k: v for k, v in res.items() if isinstance(v, (int, float, str, bool)) or v is None},
                    "score": float(res["score"])})
    return out


def cmd_score(a) -> int:
    work = Path(a.work)
    tmp = work / "tmp" / a.arm
    tmp.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fam in FAMILIES:
        g = work / "gen" / a.arm / f"{fam}.jsonl"
        if not g.exists():
            continue
        rows = read(work / "data" / fam / "eval.jsonl")
        replies = {r["id"]: r["reply"] for r in read(g) if r["sample"] == 0}
        scored = score_rows(fam, rows, replies, tmp)
        write(work / "scores" / a.arm / f"{fam}.jsonl", scored)
        summary[fam] = {"n": len(scored), "replied": len(replies), "mean": sum(s["score"] for s in scored) / len(scored)}
    (work / "scores" / a.arm / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({a.arm: summary}))
    return 0


def cmd_rft(a) -> int:
    work = Path(a.work)
    pool = read(work / "data" / a.family / "rft_pool.jsonl")
    gens = read(work / "gen" / "rft" / f"{a.family}.jsonl")
    by_id: dict[str, list[dict]] = {}
    for g in gens:
        by_id.setdefault(g["id"], []).append(g)
    keep, n_scored, n_correct = [], 0, 0
    for r in pool:
        best = None
        for g in sorted(by_id.get(r["id"], []), key=lambda g: (len(g["reply"]), g["sample"])):
            if g["finish_reason"] != "stop":
                continue
            n_scored += 1
            s = score_rows(a.family, [r], {r["id"]: g["reply"]}, work / "tmp" / "rft")[0]["score"]
            if s >= 1.0:
                n_correct += 1
                best = best or g
        if best is not None:
            keep.append({**{k: v for k, v in r.items() if k != "target"}, "split": "rft_train",
                         "target": best["reply"], "rft_sample": best["sample"]})
    random.Random(0).shuffle(keep)
    val, train = keep[: a.val], keep[a.val:]
    write(work / "data" / a.family / "rft_train.jsonl", train)
    if a.val:
        write(work / "data" / a.family / "rft_val.jsonl", [{**r, "split": "rft_val"} for r in val])
    print(json.dumps({"family": a.family, "pool": len(pool), "prompts_with_a_correct_reply": len(keep),
                      "train": len(train), "val": len(val), "finished_replies_scored": n_scored, "correct": n_correct}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--arm", required=True)
    r = sub.add_parser("rft")
    r.add_argument("--work", required=True)
    r.add_argument("--family", required=True, choices=["math", "logic"])
    r.add_argument("--val", type=int, default=0)
    a = ap.parse_args()
    return cmd_score(a) if a.cmd == "score" else cmd_rft(a)


if __name__ == "__main__":
    raise SystemExit(main())
