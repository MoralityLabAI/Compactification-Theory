"""SCH-01: deterministic, standard-library structural-comparison fixture.

The program operates on researcher annotations. It cannot discover historical
influence or establish any metaphysical hypothesis from these fixtures.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

STAGES = ("D", "T", "E")
AXES = ("time", "agency", "identity", "ultimate", "destination", "authority")
REQUIRED = {"id", "name", "source_kind", "source_url", "source_locator",
            "annotation_status", "paraphrase", "stages", "axes"}
SOURCE_KINDS = {"primary_doctrine", "primary_text_with_secondary_guide",
                "secondary_analysis", "synthetic_control"}
STATUSES = {"source_explicit", "researcher_interpretation", "synthetic"}


def validate(records: list[dict[str, Any]]) -> None:
    """Fail closed on incomplete or malformed annotations."""
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("expected a list with at least two records")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each record must be an object")
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"{record.get('id', '?')}: missing {sorted(missing)}")
        key = record["id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"duplicate or invalid record id: {key!r}")
        seen.add(key)
        if record["source_kind"] not in SOURCE_KINDS:
            raise ValueError(f"{key}: invalid source_kind")
        if record["annotation_status"] not in STATUSES:
            raise ValueError(f"{key}: invalid annotation_status")
        synthetic = record["source_kind"] == "synthetic_control"
        if synthetic != (record["annotation_status"] == "synthetic"):
            raise ValueError(f"{key}: synthetic control/status mismatch")
        if synthetic and record["source_url"] is not None:
            raise ValueError(f"{key}: synthetic controls have no historical source URL")
        if not synthetic and not (
            isinstance(record["source_url"], str)
            and record["source_url"].startswith("https://")
        ):
            raise ValueError(f"{key}: missing HTTPS evidence URL")
        for field in ("name", "source_locator", "paraphrase"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"{key}: missing {field}")
        stages = record["stages"]
        if not isinstance(stages, list) or not stages or any(
            not isinstance(x, str) or x not in STAGES for x in stages
        ):
            raise ValueError(f"{key}: malformed stage sequence")
        if stages != list(STAGES) and stages != ["D"]:
            raise ValueError(f"{key}: fixture supports [D,T,E] or [D] only")
        axes = record["axes"]
        if not isinstance(axes, dict) or set(axes) != set(AXES):
            raise ValueError(f"{key}: axes must be exactly {AXES}")
        if any(not isinstance(v, str) or not v.strip() for v in axes.values()):
            raise ValueError(f"{key}: empty or non-string axis label")


def has_triad(record: dict[str, Any]) -> bool:
    """The intentionally coarse forgetful map: retain only D -> T -> E."""
    return tuple(record["stages"]) == STAGES


def coarse_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Matching positive DTE signatures; two absent signatures do not match."""
    return has_triad(a) and has_triad(b)


def typed_agreement(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """Unweighted category agreement conditioned on a shared DTE signature."""
    if not coarse_match(a, b):
        return None
    return sum(a["axes"][key] == b["axes"][key] for key in AXES) / len(AXES)


def faithful_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Exact labeled-isomorphism test for the restricted three-node path model."""
    return coarse_match(a, b) and all(
        a["axes"][key] == b["axes"][key] for key in AXES
    )


def compare(records: list[dict[str, Any]]) -> dict[str, Any]:
    validate(records)
    comparisons = []
    for a, b in itertools.combinations(records, 2):
        score = typed_agreement(a, b)
        comparisons.append({
            "a": a["id"], "b": b["id"],
            "coarse_match": coarse_match(a, b),
            "typed_agreement": score,
            "faithful_match": faithful_match(a, b),
            "status": "synthetic_control_pair" if "synthetic_control" in (
                a["source_kind"], b["source_kind"]
            ) else "source_pair",
            "interpretive_pair": "researcher_interpretation" in (
                a["annotation_status"], b["annotation_status"]
            ),
        })
    return {
        "experiment": "SCH-01",
        "evidence_grade": "deterministic_single_coder_fixture",
        "sample_size": len(records),
        "pair_count": len(comparisons),
        "coarse_positive_pairs": sum(p["coarse_match"] for p in comparisons),
        "faithful_positive_pairs": sum(p["faithful_match"] for p in comparisons),
        "comparisons": comparisons,
        "inference_allowed": ["fixture_validation", "label-erasure counterexample"],
        "inference_excluded": ["historical independence", "empirical convergence",
                               "Bayesian evidence for metaphysics", "AI model inference"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures.json"))
    parser.add_argument("--output", type=Path, help="Optional deterministic JSON output file")
    args = parser.parse_args()
    data = json.loads(args.fixtures.read_text(encoding="utf-8"))
    result = compare(data["records"])
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
