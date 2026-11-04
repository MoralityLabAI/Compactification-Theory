"""QO-12 source-pinned paired two-step study: regenerate or fail.

First CI emits canonical full report in base64 chunks and exits because
the receipt is absent. Commit complete decoded result unchanged; then
the final-head audit verifies all 60 trajectories, per-cell streams,
certificates and source hashes byte-for-byte.

CPU only. No neural inference, cloud API or physical experiment.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from research.ingression_dynamics_v1.two_step_gap_pilot import (
    complete_report,SCENARIOS,SEEDS,POLICIES,CHECKPOINTS
)

SOURCE_FILES=(
    "research/ingression_dynamics_v1/two_step_gap_probe.py",
    "research/ingression_dynamics_v1/two_step_gap_pilot.py",
    "research/ingression_dynamics_v1/test_two_step_gap_probe.py",
    "research/ingression_dynamics_v1/TWO_STEP_CERTIFICATION.md",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/expected_gap_probe.py",
    "research/ingression_dynamics_v1/expected_gap_pilot.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "scripts/verify_two_step_gap.py",
)
DEST=ROOT/"results/two_step_gap/receipt.json"
BEGIN="__QO12_B64_BEGIN__"
END="__QO12_B64_END__"


def canonical():
    report=complete_report()
    n=len(SCENARIOS)*len(SEEDS)*len(POLICIES)
    expected={
        "trajectories":n,
        "snapshots":n*len(CHECKPOINTS),
        "queries":n*CHECKPOINTS[-1],
        "stream_replays":n,
        "two_step_planning_traces":len(SCENARIOS)*len(SEEDS)*2,
    }
    if report["run_counts"]!=expected:
        raise AssertionError("incorrect complete-pilot counts")
    if (len(report["groups"])!=len(SCENARIOS)*len(CHECKPOINTS)*len(POLICIES)
            or len(report["paired_comparisons"])!=len(SCENARIOS)*len(CHECKPOINTS)*2*(len(POLICIES)-1)
            or len(report["trial_rows"])!=n*len(CHECKPOINTS)
            or len(report["trajectory_hashes"])!=n
            or len(report["planning_receipts"])!=expected["two_step_planning_traces"]):
        raise AssertionError("missing source-bound trial records")
    if any(r["false_cert"] and r["true_rates_covered"]
           for r in report["trial_rows"]):
        raise AssertionError("false certificate on valid confidence event")
    if any(sum(c[name] for name in (
                "one_step_positive","two_step_positive",
                "fallback_bottleneck","fallback_width"))!=CHECKPOINTS[-1]
           for c in report["planning_receipts"]):
        raise AssertionError("planner step/cost mismatch")
    report["source_sha256"]={
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    report["source_refs"]={
        "qo07":"https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/ingression_dynamics_v1/ANYTIME_ACTIVE_QUOTIENT.md",
        "qo11":"https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/ingression_dynamics_v1/EXPECTED_GAP_RESULTS.md",
    }
    return (json.dumps(report,sort_keys=True,separators=(",",":"))+"\n").encode("ascii")


def emit(data):
    print(BEGIN,flush=True)
    encoded=base64.b64encode(data).decode("ascii")
    for i in range(0,len(encoded),4096):
        print(encoded[i:i+4096],flush=True)
    print(END,flush=True)


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",action="store_true")
    group.add_argument("--emit",action="store_true")
    args=parser.parse_args()
    data=canonical()
    if args.emit or (args.check and not DEST.exists()):
        emit(data)
    if args.check:
        if not DEST.exists():
            print("QO-12 missing source-bound result; commit emitted report",
                  file=sys.stderr)
            raise SystemExit(2)
        actual=DEST.read_bytes()
        if actual!=data:
            print("QO-12 receipt mismatch",file=sys.stderr)
            print("observed",hashlib.sha256(actual).hexdigest(),file=sys.stderr)
            print("expected",hashlib.sha256(data).hexdigest(),file=sys.stderr)
            raise SystemExit(3)
        print("QO-12 full paired receipt verified",
              hashlib.sha256(data).hexdigest())


if __name__=="__main__":
    main()
