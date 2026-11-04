"""QO-14: regenerate student checkpoint, held-out eval and paired pilot.

Exact source-bound compare-or-fail receipt. The initial workflow emits
Base64-encoded canonical UTF-8 JSON in 4096-character chunks and fails
if the receipt has not been committed. It then passes on the same
source-pinned branch and runner environment.

Training/test rate pairs and seeds are disjoint. Model predictions
never replace the independent QO-07 anytime-valid certificate.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.ingression_dynamics_v1.recurrent_distillation_pilot import (
    build_report,TRAIN_RATES,TEST_RATES,TRAIN_SEEDS,TEST_SEEDS,
    POLICIES,CHECKPOINTS,ROLLOUT_RATE_CASES,ROLLOUT_SEEDS
)

SOURCE_FILES=(
    "research/ingression_dynamics_v1/certificate_reachability.py",
    "research/ingression_dynamics_v1/recurrent_teacher_data.py",
    "research/ingression_dynamics_v1/recurrent_probe_student.py",
    "research/ingression_dynamics_v1/recurrent_distillation_pilot.py",
    "research/ingression_dynamics_v1/test_recurrent_distillation.py",
    "research/ingression_dynamics_v1/RECURRENT_CERTIFICATE_DISTILLATION.md",
    "scripts/verify_recurrent_distillation.py",
)
DEST=ROOT/"results/recurrent_certificate_distillation/receipt.json"


def canonical_bytes():
    report=build_report()
    expected_trials=(len(ROLLOUT_RATE_CASES)*len(ROLLOUT_SEEDS)*
                     len(POLICIES)*len(CHECKPOINTS))
    assert expected_trials==192
    assert report["counts"]=={
        "training_examples":864,"test_examples":378,
        "policy_trials":expected_trials,
        "rollout_queries":sum(CHECKPOINTS)*(
            len(ROLLOUT_RATE_CASES)*len(ROLLOUT_SEEDS)*len(POLICIES)),
    }
    assert len(report["rows"])==expected_trials
    assert len(report["groups"])==len(ROLLOUT_RATE_CASES)*len(CHECKPOINTS)*len(POLICIES)
    assert not (set(TRAIN_RATES)&set(TEST_RATES))
    assert not (set(TRAIN_SEEDS)&set(TEST_SEEDS))
    assert all(not r["false_cert"] or not r["true_rate_coverage"] for r in report["rows"])
    report["source_sha256"]={
        path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        for path in SOURCE_FILES
    }
    report["provenance"]={
        "qo13_teacher_merge":"a2905acbd47bc4443d51741c42a5c8b01ff902c8",
        "qo07_confidence":"https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/ingression_dynamics_v1/ANYTIME_ACTIVE_QUOTIENT.md",
        "model_kind":"fixed recurrent reservoir plus trained ridge readout",
        "numpy_pin":"2.3.5",
    }
    return (json.dumps(report,sort_keys=True,separators=(",",":"),
                       allow_nan=False)+"\n").encode("ascii")


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit",action="store_true")
    group.add_argument("--check",action="store_true")
    args=parser.parse_args()
    payload=canonical_bytes()
    if args.emit or (args.check and not DEST.exists()):
        print("__QO14_B64_BEGIN__",flush=True)
        encoded=base64.b64encode(payload).decode("ascii")
        for i in range(0,len(encoded),4096):
            print(encoded[i:i+4096],flush=True)
        print("__QO14_B64_END__",flush=True)
    if args.check:
        if not DEST.exists():
            raise FileNotFoundError("QO-14 receipt not yet committed")
        if DEST.read_bytes()!=payload:
            print("observed:",hashlib.sha256(DEST.read_bytes()).hexdigest())
            print("expected:",hashlib.sha256(payload).hexdigest())
            raise AssertionError("QO-14 trained checkpoint or report no longer matches sources")
        print("QO-14 trained checkpoint and all paired rollouts verified",
              hashlib.sha256(payload).hexdigest())


if __name__=="__main__":
    main()
