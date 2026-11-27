"""Replay the finite compression certificate; optionally check its Lean support lemma."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from compression import Example, decode, hypotheses, reconstruct, version_space

ROOT = Path(__file__).resolve().parent


def audit(results: Path) -> dict:
    report = json.loads((results / "report.json").read_text())
    raw = (results / "training_constraints.json").read_bytes()
    corpus = json.loads(raw)
    packet = (results / "compressed.bin").read_bytes()
    dimension, small = decode(packet)
    full = [Example(r["bits"], r["label"]) for r in corpus["constraints"]]
    selected = json.loads((results / "selected.json").read_text())
    checks = {
        "training_constraints_hash": hashlib.sha256(raw).hexdigest() == report["training_constraints_sha256"],
        "packet_hash": hashlib.sha256(packet).hexdigest() == report["packet_sha256"],
        "positive_multiplicities": all(type(r["multiplicity"]) is int and r["multiplicity"] > 0
                                       for r in corpus["constraints"]),
        "constraint_count": len(full) == report["distinct_training_constraints"],
        "source_sample_count": sum(r["multiplicity"] for r in corpus["constraints"]) == report["training_examples"],
        "root_count": len(set(corpus["root_ids"])) == len(corpus["root_ids"]) == report["training_roots"],
        "selected_values_are_source_constraints": all((x.bits, x.label) in {(v.bits, v.label) for v in full}
                                                       for x in small),
        "metadata_matches_transmitted_values": [(x.bits, x.label) for x in small] ==
            [(r["bits"], r["label"]) for r in selected["selected_examples"]],
        "nonempty_full_version_space": bool(version_space(full, dimension)),
        "same_complete_version_space": version_space(full, dimension) == version_space(small, dimension),
        "same_fixed_decoder": reconstruct(full, dimension) == reconstruct(small, dimension),
        "stored_decoder_matches": reconstruct(small, dimension)[0] == report["compressed_decoder_id"],
        "actual_packet_byte_count": len(packet) == report["transmitted_bytes"],
        "declared_hypothesis_count": len(hypotheses(dimension)) == report["hypotheses"],
        "source_hashes_match": all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
                                   for p, sha in report["source_sha256"].items()),
    }
    return {"schema": "COG07/compression-audit/v1", "checks": checks,
            "passed": all(checks.values()), "hypotheses_checked": len(hypotheses(dimension)),
            "surviving_hypotheses": version_space(full, dimension).bit_count(),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                              for p in ["compression.py", "audit.py", "FiniteCompression.lean"]}}


def lean_replay(executable: Path) -> dict:
    source = ROOT / "FiniteCompression.lean"
    run = subprocess.run([str(executable), str(source)], capture_output=True, text=True, timeout=60)
    output = run.stdout + run.stderr
    parsed = dict(re.findall(r"'COG07\.([a-z_]+)' depends on axioms: \[([^]]*)\]", output))
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    names = {"cover_preserves_consistency", "cover_preserves_version_space", "fixed_decoder_is_preserved"}
    axioms = {name: [a.strip() for a in value.split(",") if a.strip()] for name, value in parsed.items()}
    passed = (run.returncode == 0 and set(parsed) == names
              and all(set(v) <= allowed for v in axioms.values())
              and "sorryAx" not in output and "error:" not in output)
    version = subprocess.run([str(executable), "--version"], capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
    return {"passed": passed, "returncode": run.returncode, "version": version,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "lean_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "axioms": axioms,
            "scope": "General cover-implies-consistency and fixed-decoder preservation lemmas. "
                     "The concrete five-example certificate is checked by finite Python enumeration.",
            "stdout": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path)
    p.add_argument("--lean", type=Path)
    args = p.parse_args()
    report = audit(args.results)
    if args.lean:
        report["lean"] = lean_replay(args.lean)
        report["passed"] = report["passed"] and report["lean"]["passed"]
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": len(report["checks"]), "passed": report["passed"],
                      "lean": report.get("lean", {}).get("passed")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
