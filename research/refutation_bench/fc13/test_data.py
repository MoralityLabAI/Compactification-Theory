"""FC-13 fresh original MathCan theorem holdout, chosen before any score.

Exactly the first four source-order theorem declarations in each of four
unseen modules, preregistered in commit
309fed65caf67da46db8cb981757a9326bc59fbc.
All 16 original reference proofs compile under Lean4.35 as an oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from research.refutation_bench.fc11 import source_core as original
from research.refutation_bench.fc12 import train_data as fc12_train
from research.refutation_bench.fc12 import test_data as fc12_test

PREREGISTRATION="309fed65caf67da46db8cb981757a9326bc59fbc"
TEST_MODULES={
    "MathCan/HookMultiTick.lean":(
        "distribute_sound","three_account_conservation",
        "before_swap_exact_input","specified_credit_restores_debit",
    ),
    "MathCan/Settlement.lean":(
        "marked_total_conserved","exact_five_account_partition",
        "terminal_pool_floor_budget","certificate_gap_identity",
    ),
    "MathCan/TOEJordan.lean":(
        "metric_vanishes_at_double_root","mixed_vanishes_at_double_root",
        "metric_derivative_vanishes_at_root","rank_one_root_kernel",
    ),
    "MathCan/TOENonAbelian.lean":(
        "flavor_exchange","signed_color_norm",
        "signed_color_dot","mixed_under_flavor_exchange",
    ),
}


def assert_source_contract(source:Path):
    original.source_lean_version(source)
    if len(TEST_MODULES)!=4 or sum(map(len,TEST_MODULES.values()))!=16:
        raise AssertionError("FC13 original theorem count changed")
    previous=set(fc12_train.TRAIN_MODULES)|set(fc12_test.TEST_MODULES)
    if set(TEST_MODULES)&previous:
        raise AssertionError("new source module was previously FC12 train/test")
    for name,theorems in TEST_MODULES.items():
        raw=(source/name).read_text(encoding="utf8")
        declarations=re.findall(r"^theorem\s+([A-Za-z_0-9']+)",raw,re.MULTILINE)
        if tuple(declarations[:4])!=theorems:
            raise AssertionError(f"first-four original theorem selection mutated: {name}")


def emit(source:Path,out:Path):
    assert_source_contract(source)
    report=original.emit_source(source,TEST_MODULES,out,phase="fc13-preregistered-four-unseen-original-source-modules")
    if report["theorems"]!=16 or report["stages"]<16:
        raise AssertionError("expected 16 full original theorem reference scripts")
    report.update(
        experiment="FC13",
        preregistration_sha=PREREGISTRATION,
        fresh_modules_disjoint_from_FC11_and_FC12=True,
        theorem_selection="source-order first-four per module",
        heldout_original_reference_oracle_count=16,
    )
    (out/"FC13_original_source_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def extract(out:Path):
    report=original.extract_states(out)
    rows=[json.loads(x) for x in (out/"actual_goal_states.jsonl").read_text().splitlines()]
    initial={(x["source_file"],x["theorem"]) for x in rows if x["stage_index"]==0}
    expected={(path,name) for path,names in TEST_MODULES.items() for name in names}
    if initial!=expected:
        raise AssertionError("not all 16 original theorem initial goal states were captured")
    report.update(
        experiment="FC13",
        preregistration_sha=PREREGISTRATION,
        initial_genuine_single_theorem_goals=len(initial),
        test_modules=list(TEST_MODULES),
    )
    (out/"FC13_goal_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--extract",action="store_true")
    a=p.parse_args()
    result=extract(a.out) if a.extract else emit(a.source,a.out)
    print("FC13_SOURCE",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
