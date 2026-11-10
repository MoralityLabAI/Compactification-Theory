"""OIC-06 exact corruption-bound reference and deterministic replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from robust import reference

DEFAULT = Path(__file__).resolve().parent / "results" / "robust_reference.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    current = reference()
    if args.check:
        if current != json.loads(DEFAULT.read_text(encoding="utf-8")):
            raise SystemExit("OIC-06 robust replay mismatch")
        print("OIC-06 exact error/erasure replay passed")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.check:
        print(json.dumps(current, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
