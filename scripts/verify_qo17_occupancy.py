"""QO-17 deterministic full training, test and visited-calibration verifier.

First run --emit prints a complete canonical receipt between base64 markers.
Review and commit its exact bytes, then run --check on final PR head.
Both paths regenerate all models, 192 rollouts, visited-state teacher labels,
and exact policy primary metrics from source.
"""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.qo17_occupancy_pilot import (
    full_report, LEARNED_ARMS, TRAINED_ARMS,
)

DEST=ROOT/"results/qo17_occupancy/receipt.json"
BEGIN="__QO17_B64_BEGIN__"
END="__QO17_B64_END__"
SOURCES=(
    "research/ingression_dynamics_v1/QO17_OCCUPANCY_MARGIN_PROTOCOL.md",
    "research/ingression_dynamics_v1/qo17_frozen_contract.py",
    "research/ingression_dynamics_v1/qo17_margin_gate_loss.py",
    "research/ingression_dynamics_v1/qo17_occupancy_data.py",
    "research/ingression_dynamics_v1/qo17_occupancy_pilot.py",
    "research/ingression_dynamics_v1/test_qo17_occupancy.py",
    "research/ingression_dynamics_v1/tiny_recurrent_certificate_student.py",
    "research/ingression_dynamics_v1/certificate_student_data.py",
    "research/ingression_dynamics_v1/gap_student_data.py",
    "research/ingression_dynamics_v1/gap_progress_teacher.py",
    "research/ingression_dynamics_v1/gap_student_pilot.py",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/certificate_reachability.py",
    "research/ingression_dynamics_v1/expected_gap_probe.py",
    "research/ingression_dynamics_v1/two_step_gap_probe.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "research/ingression_dynamics_v1/gap_student_frozen_test.py",
    "research/ingression_dynamics_v1/certificate_boundary_test_contract.py",
    "results/gap_student/receipt.json",
    "scripts/verify_qo17_occupancy.py",
)

def canonical_bytes():
    result=full_report()
    from research.ingression_dynamics_v1.qo17_frozen_contract import (
        ARMS, FROZEN_SCENARIOS, FROZEN_POLICY_SEEDS, POLICY_CHECKPOINTS,
        FROZEN_SUPERVISED_SEEDS, MAIN_STUDENT,
    )
    n=len(ARMS)*len(FROZEN_SCENARIOS)*len(FROZEN_POLICY_SEEDS)
    expected={
        "trajectories":n,
        "snapshots":n*len(POLICY_CHECKPOINTS),
        "queries":n*POLICY_CHECKPOINTS[-1],
    }
    if result["counts"]!=expected:
        raise AssertionError("QO-17 frozen budgets or source counts drifted")
    if len(result["policy_rows"])!=expected["snapshots"] or (
            len(result["trajectory_hashes"])!=expected["trajectories"]):
        raise AssertionError("missing QO-17 full paired query receipts")
    if len(result["policy_groups"])!=len(POLICY_CHECKPOINTS)*len(ARMS):
        raise AssertionError("missing QO-17 policy budget strata")
    if not result["frozen_test_examples"]:
        raise AssertionError("QO-17 independent supervised test is empty")
    if set(result["fits"])!=set(TRAINED_ARMS) or (
            len(result["supervised_metrics"])!=len(LEARNED_ARMS)):
        raise AssertionError("QO-17 ablation source/metric rows incomplete")
    if result["primary"]["policy"]!=MAIN_STUDENT or (
            result["primary"]["budget"]!=POLICY_CHECKPOINTS[-1]):
        raise AssertionError("QO-17 prespecified primary policy changed")
    for arm,model_params in result["parameter_counts"].items():
        if ("linear" in arm and not "recurrent" in arm and model_params!=26):
            raise AssertionError("linear parameter count changed")
        if "recurrent" in arm and model_params!=138:
            raise AssertionError("recurrent model width changed")
    for arm,fit in result["fits"].items():
        if fit["best_epoch"]<1 or fit["best_epoch"]>65:
            raise AssertionError("validation-selected epoch outside source cap")
    for row in result["policy_rows"]:
        if row["false_cert"] and row["true_rates_covered"]:
            raise AssertionError("QO-07 certified wrong decision on covered truth")
    for arm,cal in result["visited_calibration"].items():
        n=cal["metrics"]["eligible_visited_samples"]
        if n!=len(cal["samples"]) or (
                sum(cal["metrics"][k] for k in (
                    "true_positive","false_positive",
                    "true_negative","false_negative"))!=n):
            raise AssertionError("visited-state calibration accounting mismatch")
    result["source_sha256"]={
        p:sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES
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
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit",action="store_true")
    mode.add_argument("--check",action="store_true")
    args=p.parse_args()
    raw=canonical_bytes()
    if args.emit:
        emit(raw)
        print("QO-17 canonical receipt sha256:",sha256(raw).hexdigest(),
              flush=True)
    if args.check:
        if not DEST.is_file():
            raise SystemExit("QO-17 receipt absent; review first full --emit and commit")
        actual=DEST.read_bytes()
        if actual!=raw:
            print("observed sha256:",sha256(actual).hexdigest(),file=sys.stderr)
            print("expected sha256:",sha256(raw).hexdigest(),file=sys.stderr)
            raise SystemExit("QO-17 source-bound weights or results differ")
        print("QO-17 source-bound reproducible receipt:",
              sha256(raw).hexdigest(),flush=True)

if __name__=="__main__":
    main()
