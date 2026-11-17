"""QO-18 source-bound complete reproducibility and paired-policy verifier.

--emit prints the complete canonical receipt between bounded base64 markers
for explicit review. Commit its exact bytes unchanged. --check retrains
both rounds and checks all held-out policy trajectories, labels, validation
gate and source hashes byte for byte on the final PR head.
"""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.qo18_dagger_pilot import (
    full_report,STUDENT_ARMS,
)

DEST=ROOT/"results/qo18_dagger/receipt.json"
BEGIN="__QO18_B64_BEGIN__"
END="__QO18_B64_END__"
SOURCES=(
    "research/ingression_dynamics_v1/QO18_DAGGER_SELECTIVE_PROTOCOL.md",
    "research/ingression_dynamics_v1/qo18_frozen_contract.py",
    "research/ingression_dynamics_v1/qo18_dagger_data.py",
    "research/ingression_dynamics_v1/qo18_gate_calibration.py",
    "research/ingression_dynamics_v1/qo18_dagger_pilot.py",
    "research/ingression_dynamics_v1/test_qo18_dagger.py",
    "research/ingression_dynamics_v1/qo17_frozen_contract.py",
    "research/ingression_dynamics_v1/qo17_margin_gate_loss.py",
    "research/ingression_dynamics_v1/qo17_occupancy_data.py",
    "research/ingression_dynamics_v1/qo17_occupancy_pilot.py",
    "research/ingression_dynamics_v1/tiny_recurrent_certificate_student.py",
    "research/ingression_dynamics_v1/certificate_student_data.py",
    "research/ingression_dynamics_v1/gap_progress_teacher.py",
    "research/ingression_dynamics_v1/gap_student_data.py",
    "research/ingression_dynamics_v1/gap_student_pilot.py",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/two_step_gap_probe.py",
    "research/ingression_dynamics_v1/expected_gap_probe.py",
    "research/ingression_dynamics_v1/certificate_reachability.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "results/qo17_occupancy/receipt.json",
    "scripts/verify_qo18_dagger.py",
)

def canonical_bytes():
    result=full_report()
    # This import follows BOTH source-only DAgger training and gate choice.
    from research.ingression_dynamics_v1.qo18_frozen_contract import (
        FROZEN_SCENARIOS,FROZEN_POLICY_SEEDS,POLICY_CHECKPOINTS,ARMS,
        PRIMARY_ARM,contract_sha256,
    )
    n=len(FROZEN_SCENARIOS)*len(FROZEN_POLICY_SEEDS)*len(ARMS)
    expected={
        "trajectories":n,
        "snapshots":n*len(POLICY_CHECKPOINTS),
        "queries":n*POLICY_CHECKPOINTS[-1],
    }
    if result["counts"]!=expected:
        raise AssertionError("QO-18 prespecified query/trajectory count drift")
    if len(result["policy_rows"])!=expected["snapshots"] or (
        len(result["trajectory_hashes"])!=expected["trajectories"]):
        raise AssertionError("QO-18 missing policy snapshots or trace hashes")
    if len(result["policy_groups"])!=len(POLICY_CHECKPOINTS)*len(ARMS):
        raise AssertionError("QO-18 missing grouped policy metrics")
    if result["test_contract_sha256"]!=contract_sha256():
        raise AssertionError("QO-18 independent test contract changed")
    if result["primary_policy"]["arm"]!=PRIMARY_ARM or (
        result["primary_policy"]["budget"]!=POLICY_CHECKPOINTS[-1]):
        raise AssertionError("QO-18 primary policy or budget changed")
    if len(result["training"]["rounds"])!=2 or (
        len(result["student_models"])!=len(STUDENT_ARMS)):
        raise AssertionError("two DAgger rounds and four student arms required")
    if any(trainable!=138 for trainable in result["parameter_counts"].values()):
        raise AssertionError("student model parameter counts drifted")
    if any(macs!=240 for macs in result["inference_macs_per_call"].values()):
        raise AssertionError("student inference MACs drifted")
    for row in result["policy_rows"]:
        if row["false_cert"] and row["true_rates_covered"]:
            raise AssertionError("false certificate on covered confidence event")
    for arm in STUDENT_ARMS:
        metrics=result["visited_engagement"][arm]["metrics"]
        if (metrics["tp"]+metrics["fp"]+metrics["tn"]+metrics["fn"]!=
            len(result["visited_engagement"][arm]["samples"])):
            raise AssertionError("post-hoc gate counts incorrect")
    threshold=result["calibration_selection"]["threshold"]
    if threshold is None:
        if result["calibration_selection"]["feasible"]:
            raise AssertionError("infeasible threshold marked valid")
        checks=result["primary_policy"]["test_gate_confusion"]
        if checks["tp"] or checks["fp"]:
            raise AssertionError("safe bottleneck fallback unexpectedly engaged")
    result["source_sha256"]={
        path:sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCES
    }
    return (json.dumps(result,sort_keys=True,separators=(",",":"),
                       allow_nan=False)+"\n").encode("ascii")

def emit(raw):
    print(BEGIN,flush=True)
    encoded=base64.b64encode(raw).decode("ascii")
    for i in range(0,len(encoded),4096):
        print(encoded[i:i+4096],flush=True)
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
        print("QO-18 canonical receipt sha256:",sha256(raw).hexdigest(),
              flush=True)
    if args.check:
        if not DEST.is_file():
            raise SystemExit("QO-18 receipt missing; run --emit, review and commit")
        actual=DEST.read_bytes()
        if actual!=raw:
            print("observed sha256:",sha256(actual).hexdigest(),file=sys.stderr)
            print("expected sha256:",sha256(raw).hexdigest(),file=sys.stderr)
            raise SystemExit("QO-18 retraining or independent result differs")
        print("QO-18 source-bound reproducible receipt:",
              sha256(raw).hexdigest(),flush=True)

if __name__=="__main__":
    main()
