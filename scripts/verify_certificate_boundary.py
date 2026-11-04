"""QO-15 deterministic source-bound training/evaluation audit.

On initial registered source, --emit writes the complete canonical receipt
to stdout in bounded base64 lines for review and commit. --check requires a
previously committed receipt and compares all bytes, including learned
weights, frozen test summaries, complete policy rows, and source hashes.
"""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.ingression_dynamics_v1.certificate_boundary_student import (
    ARMS, full_report
)
from research.ingression_dynamics_v1.certificate_boundary_test_contract import (
    FROZEN_SCENARIOS, FROZEN_POLICY_SEEDS, FROZEN_POLICY_CHECKPOINTS
)

DEST = ROOT / "results/certificate_boundary/receipt.json"
SOURCES = (
    "research/ingression_dynamics_v1/CERTIFICATE_BOUNDARY_STUDENT.md",
    "research/ingression_dynamics_v1/certificate_boundary_student.py",
    "research/ingression_dynamics_v1/certificate_boundary_test_contract.py",
    "research/ingression_dynamics_v1/test_certificate_boundary_student.py",
    "research/ingression_dynamics_v1/certificate_student_data.py",
    "research/ingression_dynamics_v1/tiny_recurrent_certificate_student.py",
    "research/ingression_dynamics_v1/certificate_student_pilot.py",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/certificate_reachability.py",
    "research/ingression_dynamics_v1/two_step_gap_probe.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "results/certificate_student/receipt.json",
    "scripts/verify_certificate_boundary.py",
)
BEGIN = "__QO15_B64_BEGIN__"
END = "__QO15_B64_END__"


def canonical_bytes():
    result = full_report()
    n = len(FROZEN_SCENARIOS) * len(FROZEN_POLICY_SEEDS) * len(ARMS)
    expected = {
        "trajectories": n,
        "snapshots": n * len(FROZEN_POLICY_CHECKPOINTS),
        "queries": n * FROZEN_POLICY_CHECKPOINTS[-1],
    }
    if result["counts"] != expected:
        raise AssertionError("frozen paired evaluation counts drifted")
    if len(result["policy_rows"]) != expected["snapshots"]:
        raise AssertionError("frozen policy snapshots incomplete")
    if len(result["trajectory_hashes"]) != n:
        raise AssertionError("frozen policy stream receipts incomplete")
    if any(row["false_cert"] and row["true_rates_covered"]
           for row in result["policy_rows"]):
        raise AssertionError("wrong confidence-covered certificate")
    if result["training"]["split"] != "train" or (
            result["validation"]["split"] != "validation"):
        raise AssertionError("frontier split provenance mismatch")
    result["source_sha256"] = {
        source: sha256((ROOT / source).read_bytes()).hexdigest()
        for source in SOURCES
    }
    return (json.dumps(result, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")


def emit(payload):
    print(BEGIN, flush=True)
    encoded = base64.b64encode(payload).decode("ascii")
    for i in range(0, len(encoded), 4096):
        print(encoded[i:i+4096], flush=True)
    print(END, flush=True)


def main():
    parser = argparse.ArgumentParser()
    opts = parser.add_mutually_exclusive_group(required=True)
    opts.add_argument("--emit", action="store_true")
    opts.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = canonical_bytes()
    if args.emit:
        emit(expected)
        print("QO-15 complete deterministic receipt sha256:",
              sha256(expected).hexdigest(), flush=True)
    if args.check:
        if not DEST.is_file():
            raise SystemExit("QO-15 receipt missing: run --emit, review, commit")
        observed = DEST.read_bytes()
        if observed != expected:
            print("observed_sha256:", sha256(observed).hexdigest(),
                  file=sys.stderr)
            print("expected_sha256:", sha256(expected).hexdigest(),
                  file=sys.stderr)
            raise SystemExit("QO-15 source-bound receipt differs")
        print("QO-15 certified reproducible receipt sha256:",
              sha256(expected).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
