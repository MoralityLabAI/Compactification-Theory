"""QO-10: full-source-bound, paired-stream, compare-or-fail pilot receipt.

The first CI execution emits the precise canonical JSON in 4096-character
BASE64 chunks and fails because the committed receipt is absent. Review
that output, commit it verbatim, and rerun --check to verify every result.

No network/model/GPU calls; only local synthetic Bernoulli simulation.
"""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.decision_boundary_probe import (
    full_report,SCENARIOS,SEEDS,POLICIES,CHECKPOINTS
)

SOURCES=(
    "research/ingression_dynamics_v1/decision_boundary_probe.py",
    "research/ingression_dynamics_v1/test_decision_boundary_probe.py",
    "research/ingression_dynamics_v1/DECISION_BOUNDARY_PILOT.md",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "research/ingression_dynamics_v1/fractional_cover_probe.py",
    "research/ingression_dynamics_v1/geometric_quotient.py",
    "research/ingression_dynamics_v1/rate_sampling.py",
    "scripts/verify_decision_boundary_probe.py",
)
DEST=ROOT/"results/decision_boundary_probe/receipt.json"
BEGIN="__QO10_B64_BEGIN__"
END="__QO10_B64_END__"


def canonical():
    report=full_report()
    expected=len(SCENARIOS)*len(SEEDS)*len(POLICIES)
    queries=expected*CHECKPOINTS[-1]
    snapshots=expected*len(CHECKPOINTS)
    if report["run_counts"]!={
        "trajectories":expected,
        "snapshots":snapshots,
        "queries":queries,
        "stream_replays":expected,
    }:
        raise AssertionError("wrong trajectory or query count")
    if (len(report["trajectory_receipt_hashes"])!=expected
            or len(report["trial_rows"])!=snapshots
            or len(report["groups"])!=len(SCENARIOS)*len(CHECKPOINTS)*len(POLICIES)
            or len(report["paired_comparisons"])!=len(SCENARIOS)*len(CHECKPOINTS)*(len(POLICIES)-1)):
        raise AssertionError("incomplete pilot report")
    if any(x["false_cert"] and x["true_rates_covered"]
           for x in report["trial_rows"]):
        raise AssertionError("false certificate on covered true model")
    report["source_sha256"]={
        path:sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES
    }
    report["reference_git_heads"]={
        "evalmath_qi04_pr":"https://github.com/MoralityLabAI/EvalMath/pull/27",
        "evalmath_qi03_merge":"6926d9610030b09ef04315f5203e608daed3fc45",
        "ontological_mor08_merge":"0ca389c472eb80a55108ca05e19ac8c3a00ac427",
        "qo08_merge":"ca68f00e1f1cc8c04cb3f889002a4bb069f4db88",
        "qo09_merge":"634318980cfd3f8f92dcf93fdeb0bd771cbd0647",
    }
    return (json.dumps(report,sort_keys=True,separators=(",",":"))+"\n").encode("ascii")


def emit(encoded):
    print(BEGIN,flush=True)
    chunks=base64.b64encode(encoded).decode("ascii")
    for i in range(0,len(chunks),4096):
        print(chunks[i:i+4096],flush=True)
    print(END,flush=True)


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--emit",action="store_true")
    modes.add_argument("--check",action="store_true")
    args=parser.parse_args()
    encoded=canonical()
    if args.emit or (args.check and not DEST.is_file()):
        emit(encoded)
    if args.check:
        if not DEST.is_file():
            print("QO-10 result missing; commit the emitted byte-for-byte canonical JSON.",
                  file=sys.stderr)
            raise SystemExit(2)
        observed=DEST.read_bytes()
        if observed!=encoded:
            print("QO-10 source or trial receipt changed.",file=sys.stderr)
            print("observed sha256:",sha256(observed).hexdigest(),file=sys.stderr)
            print("expected sha256:",sha256(encoded).hexdigest(),file=sys.stderr)
            raise SystemExit(3)
        print("QO-10 full fixed-protocol matched-stream receipt verified:",
              sha256(encoded).hexdigest())


if __name__=="__main__":
    main()
