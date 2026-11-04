"""QO-16 source-bound deterministic training and held-out policy verifier.

--emit: produce a canonical complete receipt, bounded-line base64, for review.
--check: retrain both students and all 96 policies and compare receipt bytes.
All generation is CPU-only using the Python standard library.
"""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.gap_student_pilot import (
    full_report, ARMS,
)

DEST=ROOT/"results/gap_student/receipt.json"
BEGIN="__QO16_B64_BEGIN__"
END="__QO16_B64_END__"
SOURCES=(
    "research/ingression_dynamics_v1/DENSE_GAP_STUDENT_PROTOCOL.md",
    "research/ingression_dynamics_v1/gap_student_frozen_test.py",
    "research/ingression_dynamics_v1/gap_progress_teacher.py",
    "research/ingression_dynamics_v1/gap_student_data.py",
    "research/ingression_dynamics_v1/gap_student_pilot.py",
    "research/ingression_dynamics_v1/test_gap_student.py",
    "research/ingression_dynamics_v1/tiny_recurrent_certificate_student.py",
    "research/ingression_dynamics_v1/certificate_student_data.py",
    "research/ingression_dynamics_v1/certificate_boundary_test_contract.py",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/expected_gap_probe.py",
    "research/ingression_dynamics_v1/two_step_gap_probe.py",
    "research/ingression_dynamics_v1/certificate_reachability.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "scripts/verify_gap_student.py",
)

def canonical_bytes():
    receipt=full_report()
    # Import fixed test contract only after models have been fitted.
    from research.ingression_dynamics_v1.gap_student_frozen_test import (
        FROZEN_SCENARIOS,FROZEN_POLICY_SEEDS,
        FROZEN_POLICY_CHECKPOINTS,
    )
    n=len(FROZEN_SCENARIOS)*len(FROZEN_POLICY_SEEDS)*len(ARMS)
    counts={
        "trajectories":n,
        "snapshots":n*len(FROZEN_POLICY_CHECKPOINTS),
        "queries":n*FROZEN_POLICY_CHECKPOINTS[-1],
    }
    if receipt["counts"]!=counts:
        raise AssertionError("QO-16 fixed paired budget drifted")
    if len(receipt["trajectory_hashes"])!=counts["trajectories"] or (
            len(receipt["policy_rows"])!=counts["snapshots"]):
        raise AssertionError("QO-16 incomplete policy receipts")
    if receipt["model_parameters"]!={"linear_gap_student":26,
                                      "recurrent_gap_student":138}:
        raise AssertionError("QO-16 model parameter count changed")
    if receipt["model_macs_per_inference"]!={"linear_gap_student":24,
                                              "recurrent_gap_student":240}:
        raise AssertionError("QO-16 model computation changed")
    for row in receipt["policy_rows"]:
        if row["false_cert"] and row["true_rates_covered"]:
            raise AssertionError("incorrect certificate on covered truth")
    for arm,fit in receipt["training_receipts"].items():
        if fit["parameters"] not in (26,138) or not (
                1<=fit["best_epoch"]<=65):
            raise AssertionError("bad validation-selected model")
    receipt["source_sha256"]={
        p:sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES
    }
    return (json.dumps(receipt,sort_keys=True,separators=(",",":"),
                       allow_nan=False)+"\n").encode("ascii")

def emit(raw):
    print(BEGIN,flush=True)
    v=base64.b64encode(raw).decode("ascii")
    for pos in range(0,len(v),4096):
        print(v[pos:pos+4096],flush=True)
    print(END,flush=True)

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit",action="store_true")
    mode.add_argument("--check",action="store_true")
    args=parser.parse_args()
    raw=canonical_bytes()
    if args.emit:
        emit(raw)
        print("QO-16 canonical receipt sha256:",sha256(raw).hexdigest(),
              flush=True)
    if args.check:
        if not DEST.is_file():
            raise SystemExit("QO-16 receipt missing; emit, review, commit")
        actual=DEST.read_bytes()
        if actual!=raw:
            print("observed:",sha256(actual).hexdigest(),file=sys.stderr)
            print("expected:",sha256(raw).hexdigest(),file=sys.stderr)
            raise SystemExit("QO-16 receipt differs from source-bound rerun")
        print("QO-16 source-bound verified:",sha256(raw).hexdigest(),
              flush=True)

if __name__=="__main__":
    main()
