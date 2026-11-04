"""QO-09 deterministic, source-SHA-256-bound five-policy trial receipt.

Usage: --check verifies the exact committed receipt; --emit prints it for
initial review. On a missing receipt, --check emits the expected complete
JSON between markers before failing so its bytes can be committed.

The checker re-executes all 150 fixed policy trajectories, verifies paired
stream prefixes, and checks every rational LP primal/dual certificate.
No API, GPU, external model, or physical experiment calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.ingression_dynamics_v1.fractional_cover_probe import (
    build_pilot_report, SCENARIOS, SEEDS, POLICIES, CHECKPOINTS
)

SOURCES = (
    "research/ingression_dynamics_v1/fractional_cover_probe.py",
    "research/ingression_dynamics_v1/test_fractional_cover_probe.py",
    "research/ingression_dynamics_v1/FRACTIONAL_PROBE_PILOT.md",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/geometric_quotient.py",
    "research/ingression_dynamics_v1/rate_sampling.py",
    "scripts/verify_fractional_probe_pilot.py",
)
DESTINATION = ROOT / "results/fractional_probe_pilot/receipt.json"
BEGIN = "__QO09_RECEIPT_BEGIN__"
END = "__QO09_RECEIPT_END__"


def canonical_bytes():
    report = build_pilot_report()
    expected_trajectories = len(SCENARIOS) * len(SEEDS) * len(POLICIES)
    expected_queries = expected_trajectories * CHECKPOINTS[-1]
    expected_snapshots = expected_trajectories * len(CHECKPOINTS)
    assert report["run_counts"] == {
        "trajectories": expected_trajectories,
        "snapshots": expected_snapshots,
        "queries": expected_queries,
        "paired_stream_verifications": expected_trajectories,
    }
    assert len(report["trajectory_receipt_hashes"]) == expected_trajectories
    assert len(report["trial_rows"]) == expected_snapshots
    assert len(report["groups"]) == (
        len(SCENARIOS) * len(CHECKPOINTS) * len(POLICIES))
    assert len(report["paired_comparisons"]) == (
        len(SCENARIOS) * len(CHECKPOINTS) * (len(POLICIES) - 1))
    assert all(not row["false_cert"] or not row["true_rates_covered"]
               for row in report["trial_rows"])
    report["sources_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCES
    }
    report["reference_provenance"] = {
        "qi03_evalmath_merge": "6926d9610030b09ef04315f5203e608daed3fc45",
        "mor08_lean_duality_merge": "0ca389c472eb80a55108ca05e19ac8c3a00ac427",
        "qo08_harness_merge": "ca68f00e1f1cc8c04cb3f889002a4bb069f4db88",
    }
    return (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = canonical_bytes()
    if args.emit or (args.check and not DESTINATION.is_file()):
        print(BEGIN)
        sys.stdout.flush()
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        print(END, flush=True)
    if args.check:
        if not DESTINATION.is_file():
            print("QO-09 receipt missing; commit emitted source-bound JSON.", file=sys.stderr)
            raise SystemExit(2)
        observed = DESTINATION.read_bytes()
        if observed != data:
            print("QO-09 source-bound result mismatch.", file=sys.stderr)
            print("observed sha256", hashlib.sha256(observed).hexdigest(),file=sys.stderr)
            print("expected sha256", hashlib.sha256(data).hexdigest(),file=sys.stderr)
            raise SystemExit(3)
        print("QO-09 exact source-bound matched-stream receipt verified:",
              hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
