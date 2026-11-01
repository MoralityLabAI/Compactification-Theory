"""Write the G1 receipt: scenario suite, matched baseline and compaction results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

from compaction import analyse
from g1_eval import ARCHITECTURES, FAMILIES, matched_comparison, run_family

HERE = Path(__file__).parent
SOURCES = ("council.py", "g1_eval.py", "compaction.py", "finite_model.py", "g2_harness.py",
           "test_council.py", "test_compaction.py", "test_g2_harness.py", "run_g1.py")


def main(out: Path = HERE / "results" / "g1_v1.json") -> dict:
    suite = [run_family(f, a) for f in FAMILIES for a in ARCHITECTURES]
    result = {
        "schema": "6gw.g1.v1",
        "status": "offline_deterministic_simulation_only",
        "environment": f"Python {platform.python_version()}, standard library, CPU",
        "source_sha256": {s: hashlib.sha256((HERE / s).read_bytes()).hexdigest() for s in SOURCES},
        "scenario_suite": suite,
        "suite_summary": {
            arch: {
                "families": sum(r["architecture"] == arch for r in suite),
                "match_expected": sum(r["matches_expected"] for r in suite if r["architecture"] == arch),
                "match_ground_truth": sum(r["matches_ground_truth"] for r in suite if r["architecture"] == arch),
                "replay_ok": sum(r["replay_ok"] for r in suite if r["architecture"] == arch),
                "approvals": sum(r["approvals"] for r in suite if r["architecture"] == arch),
                "audit_closed": sum(r["audit_closed"] for r in suite if r["architecture"] == arch),
            } for arch in ARCHITECTURES
        },
        "matched_comparison": matched_comparison(),
        "compaction": analyse(),
        "claims_excluded": ["model calls", "training", "GPU", "networking", "real cryptographic identity",
                            "stochastic agents", "human study", "CI pass"],
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["suite_summary"], indent=1))
    for row in r["compaction"]["cases"]:
        print(row["alphabet"], row["observation"], row["exact_classes"],
              {k: v["exact"] for k, v in row["codes"].items()}, file=sys.stdout)
