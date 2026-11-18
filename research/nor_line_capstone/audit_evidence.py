"""Read-only byte audit for the NOR capstone's historical evidence ledger.

This checks the documented source files. It neither generates timings nor
replaces any original registered decision. Use each study's own archived
auditor to reconstruct its detailed arithmetic and gate checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(repository: Path, ledger_path: Path) -> dict:
    repository = repository.resolve()
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    if ledger.get("schema") != "nor.capstone.evidence-ledger.v1":
        raise ValueError("unsupported capstone evidence ledger")
    checks = []
    studies = set()
    seen_files = set()
    for study in ledger["studies"]:
        label = study["study"]
        if label in studies:
            raise ValueError(f"duplicate study: {label}")
        studies.add(label)
        for item in study["files"]:
            relative = item["path"]
            target = (repository / relative).resolve()
            if not target.is_relative_to(repository) or Path(relative).is_absolute():
                raise ValueError(f"evidence path escapes repository: {relative}")
            if relative in seen_files:
                raise ValueError(f"duplicate evidence path: {relative}")
            seen_files.add(relative)
            raw = target.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            good = len(raw) == item["bytes"] and digest == item["sha256"]
            checks.append({"study": label, "path": relative, "bytes": len(raw),
                           "sha256": digest, "matches_ledger": good})
    return {
        "schema": "nor.capstone.evidence-audit.v1",
        "historical_commit": ledger["historical_commit"],
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "study_count": len(studies),
        "file_count": len(checks),
        "all_file_hashes_match": bool(checks) and all(c["matches_ledger"] for c in checks),
        "scope": "read-only evidence bytes; original timings and verdicts preserved",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ledger_path = Path(__file__).with_name("evidence_ledger.json")
    result = audit(args.repository, ledger_path)
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data)
    else:
        print(data, end="")
    return 0 if result["all_file_hashes_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
