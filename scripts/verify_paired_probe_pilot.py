"""QO-08 deterministic source-hashed paired probe report verifier.

No network, GPU, model, or laboratory calls. A missing or stale committed
receipt causes a nonzero exit. --emit is used to retrieve the exact expected
UTF-8 JSON from a failed first CI run; --check is mandatory for merge.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.ingression_dynamics_v1.paired_probe_pilot import build_report

SOURCES = (
    "research/ingression_dynamics_v1/anytime_active_quotient.py",
    "research/ingression_dynamics_v1/paired_probe_pilot.py",
    "research/ingression_dynamics_v1/test_paired_probe_pilot.py",
    "research/ingression_dynamics_v1/PAIRED_PROBE_PILOT.md",
    "scripts/verify_paired_probe_pilot.py",
)
DEST = ROOT / "results/paired_probe_pilot/receipt.json"

BEGIN = "__QO08_RECEIPT_BEGIN__"
END = "__QO08_RECEIPT_END__"


def report_bytes():
    report = build_report()
    report["source_sha256"] = {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in SOURCES
    }
    return (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true",
                      help="emit canonical proposed receipt for review")
    mode.add_argument("--check", action="store_true",
                      help="compare exact committed receipt and fail on differences")
    args = parser.parse_args()
    encoded = report_bytes()
    if args.emit or (args.check and not DEST.exists()):
        print(BEGIN)
        sys.stdout.flush()
        sys.stdout.buffer.write(encoded)
        sys.stdout.flush()
        print(END)
    if args.check:
        if not DEST.exists():
            raise FileNotFoundError("QO-08 source-bound receipt missing: commit emitted JSON")
        if DEST.read_bytes() != encoded:
            current_hash = hashlib.sha256(DEST.read_bytes()).hexdigest()
            expected_hash = hashlib.sha256(encoded).hexdigest()
            raise AssertionError(
                "QO-08 receipt/source mismatch: observed sha256="
                + current_hash + " expected=" + expected_hash)
        print("QO-08 receipt matches current source and all paired outcomes.")
        print("sha256:", hashlib.sha256(encoded).hexdigest())


if __name__ == "__main__":
    main()
