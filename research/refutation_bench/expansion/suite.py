"""Run the public source-audit expansion alongside frozen RefutationBench v0.1.

The original seed and its hash-pinned baseline receipt remain unchanged. This
module loads a separate, publicly labelled development pack and uses the same
schema validation, scoring and input-only export contract as the base suite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import bench

HERE = Path(__file__).resolve().parent


def load_combined() -> tuple[list[dict], list[dict], dict]:
    cases, gold, sources = bench.load_dataset()
    cases = cases + bench.read_jsonl(HERE / "cases.jsonl")
    gold = gold + bench.read_jsonl(HERE / "gold.jsonl")
    bench.validate_dataset(cases, gold, sources)
    return cases, gold, sources


def ansari_witness() -> dict:
    """Exact first-block check of the identity used in Lemma 3.1.

    On its n=1 slice the unfiltered F'_1 has all base-3 digit pairs and A'
    removes just (2,2), whereas F_2 has only digits {0,1}.
    """
    def in_f2(k: int) -> bool:
        return 0 <= k < 9 and (k % 3 in (0, 1)) and (k // 3 in (0, 1))

    def in_fprime_minus_a(k: int) -> bool:
        return 0 <= k < 9 and k != 8

    k = 2
    assert in_fprime_minus_a(k) and not in_f2(k)
    lhs = [4 * t + 3 for t in range(9) if in_fprime_minus_a(t)]
    rhs = [4 * t + 3 for t in range(9) if in_f2(t)]
    return {"witness": 4 * k + 3, "in_Fprime1_minus_Aprime": True,
            "in_F2": False, "lhs": lhs, "rhs": rhs}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RefutationBench sourced expansion")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("witness")
    export = sub.add_parser("export")
    export.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    export.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cases, gold, sources = load_combined()
    if args.command == "validate":
        print(json.dumps(bench.validate_dataset(cases, gold, sources), indent=2))
        return
    if args.command == "witness":
        print(json.dumps(ansari_witness(), indent=2))
        return

    if args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]
    if not cases:
        raise bench.DataError("No cases in selected split")
    ids = {c["case_id"] for c in cases}
    if args.command == "export":
        bench.write_jsonl(args.out, [bench.prompt_record(c) for c in cases])
    elif args.command == "evaluate":
        responses = bench.read_jsonl(args.responses)
        result = bench.evaluate(cases, [g for g in gold if g["case_id"] in ids], responses)
        bench.write_json(args.out, result)
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
