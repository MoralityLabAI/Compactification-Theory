"""OIC-01 benchmark and canonical JSON replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import full_reference


HERE = Path(__file__).resolve().parent
DEFAULT_RECEIPT = HERE / "results" / "reference.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with pinned JSON receipt")
    parser.add_argument("--out", type=Path, default=None, help="write deterministic JSON receipt")
    args = parser.parse_args()
    observed = full_reference()
    if args.check:
        expected = json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("OIC-01 REPLAY MISMATCH")
        print("OIC-01 exact reference matches pinned receipt")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.check:
        print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
