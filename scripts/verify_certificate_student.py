"""QO-14 deterministic, source-bound full training and evaluation verifier.

The first --check on a missing receipt emits the exact canonical report
between base64 markers and fails. Review that full result, commit it
unchanged, and rerun --check on the final branch head. All trained
weights and policy trajectories are regenerated from public source.

No external APIs, paid GPU, torch, numpy, network or model downloads.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.certificate_student_pilot import (
    full_report,EVAL_SCENARIOS,POLICIES,POLICY_SEEDS,CHECKPOINTS
)

SOURCES=(
    "research/ingression_dynamics_v1/certificate_student_data.py",
    "research/ingression_dynamics_v1/tiny_recurrent_certificate_student.py",
    "research/ingression_dynamics_v1/certificate_student_pilot.py",
    "research/ingression_dynamics_v1/test_certificate_student.py",
    "research/ingression_dynamics_v1/CERTIFICATE_DISTILLATION.md",
    "research/ingression_dynamics_v1/certificate_reachability.py",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/two_step_gap_probe.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "scripts/verify_certificate_student.py",
)
DEST=ROOT/"results/certificate_student/receipt.json"
BEGIN="__QO14_B64_BEGIN__"
END="__QO14_B64_END__"


def canonical_bytes():
    result=full_report()
    n=len(EVAL_SCENARIOS)*len(POLICY_SEEDS)*len(POLICIES)
    expected={
        "trajectories":n,
        "snapshots":n*len(CHECKPOINTS),
        "queries":n*CHECKPOINTS[-1],
    }
    if result["counts"]!=expected:
        raise AssertionError("training/eval report missing trajectories")
    if len(result["groups"])!=len(EVAL_SCENARIOS)*len(CHECKPOINTS)*len(POLICIES):
        raise AssertionError("incomplete per-policy groups")
    if len(result["trial_rows"])!=expected["snapshots"] or (
            len(result["trajectory_hashes"])!=n):
        raise AssertionError("missing per-seed evidence")
    if result["inference_parameter_counts"]!={
            "linear_student":26,"recurrent_student":138}:
        raise AssertionError("trained parameter count changed")
    if result["inference_multiply_accumulates"]!={
            "linear_student":24,"recurrent_student":240}:
        raise AssertionError("student deployment compute changed")
    if any(r["false_cert"] and r.get("true_rates_covered",True)
           for r in result["trial_rows"]):
        raise AssertionError("wrong certificate on covered truth")
    result["source_sha256"]={
        source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        for source in SOURCES
    }
    result["mathematical_reference"]={
        "qo13_teacher":"https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/ingression_dynamics_v1/certificate_reachability.py",
        "qo07_anytime":"https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/ingression_dynamics_v1/ANYTIME_ACTIVE_QUOTIENT.md",
    }
    return (json.dumps(result,sort_keys=True,separators=(",",":"),
                       allow_nan=False)+"\n").encode("ascii")


def emit(payload):
    print(BEGIN,flush=True)
    data=base64.b64encode(payload).decode("ascii")
    for i in range(0,len(data),4096):
        print(data[i:i+4096],flush=True)
    print(END,flush=True)


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--emit",action="store_true")
    args=parser.parse_args()
    expected=canonical_bytes()
    if args.emit or (args.check and not DEST.is_file()):
        emit(expected)
    if args.check:
        if not DEST.is_file():
            print("QO-14 trained-student receipt missing; commit complete emitted report",file=sys.stderr)
            raise SystemExit(2)
        actual=DEST.read_bytes()
        if actual!=expected:
            print("QO-14 training weights or results differ from sources",file=sys.stderr)
            print("observed:",hashlib.sha256(actual).hexdigest(),file=sys.stderr)
            print("expected:",hashlib.sha256(expected).hexdigest(),file=sys.stderr)
            raise SystemExit(3)
        print("QO-14 exact regenerated model and heldout policy receipt",
              hashlib.sha256(expected).hexdigest())


if __name__=="__main__":
    main()
