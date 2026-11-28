"""End-to-end CPU smoke: every stage on a tiny random Qwen3.5, no checkpoint.

    python -m research.algernon.algebraic_reasoning_gym.arg02.smoke --run-dir D

Runs export (2 cases per split), dev shots, select, dev blocks, select,
test, test --model ff, report and replay. Scores are meaningless; the point
is that every arm, capture, statistic and receipt path executes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import analyze, runner


def run(run_dir, sample=2, profile="qwen"):
    run_dir = Path(run_dir)
    out = run_dir / "results"
    common = ["--run-dir", str(run_dir), "--smoke", "--profile", profile]
    analyze.main(["export", "--run-dir", str(run_dir), "--sample", str(sample), "--profile", profile])
    runner.main(["dev", "--stage", "shots"] + common)
    analyze.main(["select", "--run-dir", str(run_dir), "--stage", "shots"])
    runner.main(["dev", "--stage", "blocks"] + common)
    analyze.main(["select", "--run-dir", str(run_dir), "--stage", "blocks"])
    runner.main(["test"] + common)
    if profile == "qwen":
        runner.main(["test", "--model", "ff"] + common)
    analyze.main(["report", "--run-dir", str(run_dir), "--out", str(out)])
    analyze.main(["replay", "--tape", str(out / "arg02_answer_tape.jsonl.gz"),
                  "--check", str(out / "arg02_receipt.json")])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--sample", type=int, default=2)
    p.add_argument("--profile", choices=["qwen", "nanbeige"], default="qwen")
    args = p.parse_args()
    print("smoke results in", run(args.run_dir, args.sample, args.profile))


if __name__ == "__main__":
    main()
