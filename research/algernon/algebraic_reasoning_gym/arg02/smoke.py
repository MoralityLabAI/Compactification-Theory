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


def run(run_dir, sample=2):
    run_dir = Path(run_dir)
    out = run_dir / "results"
    analyze.main(["export", "--run-dir", str(run_dir), "--sample", str(sample)])
    runner.main(["dev", "--run-dir", str(run_dir), "--stage", "shots", "--smoke"])
    analyze.main(["select", "--run-dir", str(run_dir), "--stage", "shots"])
    runner.main(["dev", "--run-dir", str(run_dir), "--stage", "blocks", "--smoke"])
    analyze.main(["select", "--run-dir", str(run_dir), "--stage", "blocks"])
    runner.main(["test", "--run-dir", str(run_dir), "--smoke"])
    runner.main(["test", "--run-dir", str(run_dir), "--smoke", "--model", "ff"])
    analyze.main(["report", "--run-dir", str(run_dir), "--out", str(out)])
    analyze.main(["replay", "--tape", str(out / "arg02_answer_tape.jsonl.gz"),
                  "--check", str(out / "arg02_receipt.json")])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--sample", type=int, default=2)
    args = p.parse_args()
    print("smoke results in", run(args.run_dir, args.sample))


if __name__ == "__main__":
    main()
