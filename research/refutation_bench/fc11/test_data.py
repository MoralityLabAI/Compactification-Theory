"""FC-11 held-out ORIGINAL MathCan theorem trajectories.

This module is committed AFTER the frozen train-only source revision
2e7e749d4c7f8610a4083f4f8551d084102ea55a. The source repo SHA and
Lean toolchain are identical, while selected theorem families/files
remain disjoint. The original reference scripts are compiled only as
source certification and are NEVER an evaluation model input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import source_core as core
from . import train_data

TEST_SOURCE={
    "MathCan/Intervention.lean":(
        "obsEqOn_refl",
        "obsEqOn_symm",
        "obsEqOn_trans",
        "obsEqOn_action_subset",
        "obsEqOn_postcompose",
        "obsEqOn_action_stable",
    ),
    "MathCan/HookState.lean":(
        "closed_gauge_invariant",
        "pathSum_potential_bound",
        "closed_path_nonpositive",
        "two_order_defect_gauge",
    ),
    "MathCan/GraphAuthorization.lean":(
        "repair_budget_equiv",
        "repair_budget_mono",
        "service_iff_waste_within_budget",
        "avoidable_regret_iff_excess_slack",
    ),
}


def assert_family_disjoint():
    train=set(train_data.TRAIN_SOURCE)
    test=set(TEST_SOURCE)
    if train.intersection(test):
        raise AssertionError("train and held-out imported modules overlap")
    if len(test)!=3 or sum(len(v) for v in TEST_SOURCE.values())!=14:
        raise AssertionError("registered held-out theorem count changed")


def emit(source:Path,out:Path)->dict:
    assert_family_disjoint()
    report=core.emit_source(source,TEST_SOURCE,out,phase="test-after-train-freeze")
    if report["theorems"]!=14 or report["stages"]<20:
        raise AssertionError("missing registered test theorem tactic stages")
    report["train_only_source_commit"]="2e7e749d4c7f8610a4083f4f8551d084102ea55a"
    report["test_modules_disjoint_from_training"]=True
    (out/"test_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def extract(out:Path)->dict:
    report=core.extract_states(out)
    if report["real_compiler_states"]<20:
        raise AssertionError("too few compiler-backed held-out states")
    (out/"test_extraction.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--extract",action="store_true")
    a=p.parse_args()
    report=extract(a.out) if a.extract else emit(a.source,a.out)
    print(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
