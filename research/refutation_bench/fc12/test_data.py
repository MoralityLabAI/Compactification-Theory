"""FC-12 freshly held-out original MathCan theorems with parameterized Lean goals.

Added AFTER immutable train-only revision
8d192f0b323eade0fc764611f420414d70a0d81b, which passed
961 actual Lean tactic candidate tests and contains no heldout selector.
These four modules did not enter the FC-10/FC-11 neural tests.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from research.refutation_bench.fc11 import source_core as original
from . import core,train_data

TEST_MODULES={
    "MathCan/Hodge.lean":(
        "endpoint_append",
        "pathSum_append",
        "closed_path_gauge",
        "edge_residual_gauge",
    ),
    "MathCan/TOEDressedMatter.lean":(
        "spectral_multiplicity",
        "flrw_temporal_tracefree_iff",
        "metric_vacuum_shift",
        "gr_lapse_limit",
    ),
    "MathCan/TOELorentzBivector.lean":(
        "diagonal_matching_iff",
        "negative_discriminant_of_positive_mixed_square",
        "matched_real_spectrum_forces_zero_mixing",
        "nonzero_raised_determinant",
    ),
    "MathCan/TOETensorCompletion.lean":(
        "positivePart_nonnegative",
        "mismatched_inverse_response",
        "zero_gap_of_equal_products",
        "homogeneous_scalar_wedge_zero",
    ),
}


def check_selection():
    if len(TEST_MODULES)!=4 or sum(len(x) for x in TEST_MODULES.values())!=16:
        raise AssertionError("registered test family or theorem count changed")
    if set(TEST_MODULES)&set(train_data.TRAIN_MODULES):
        raise AssertionError("training original source files overlap heldout modules")


def emit(source:Path,out:Path)->dict:
    check_selection()
    report=original.emit_source(source,TEST_MODULES,out,phase="fc12-heldout-after-train-freeze")
    if report["theorems"]!=16 or report["stages"]<20:
        raise AssertionError("too few real original theorem stages")
    report.update({
      "experiment":"FC12",
      "train_only_source_commit":"8d192f0b323eade0fc764611f420414d70a0d81b",
      "original_source_commit":core.SOURCE_COMMIT,
      "train_and_test_modules_disjoint":True,
      "train_typed_effect_sha256":"bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec",
      "pretest_action_coverage_all_four_mandatory":True,
    })
    (out/"FC12_test_source_receipt.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def extract(out:Path)->dict:
    report=original.extract_states(out)
    rows=core.states_from_compiler(out)
    initial={(x["source_file"],x["theorem"]) for x in rows if x["stage_index"]==0}
    expected={(p,name) for p,names in TEST_MODULES.items() for name in names}
    if initial!=expected:
        raise AssertionError("missing original theorem initial Lean goal state")
    report["heldout_initial_goals"]=len(initial)
    report["heldout_distinct_research_modules"]=len(TEST_MODULES)
    (out/"FC12_test_goal_receipt.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--extract",action="store_true")
    args=p.parse_args()
    out=extract(args.out) if args.extract else emit(args.source,args.out)
    print(json.dumps(out,sort_keys=True,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
