"""FC-12 TRAIN-ONLY theorem selector from original MathCan Lean4.35 sources.

The train selector precedes the future FC-12 test_data.py module. No held-out
source module may be opened to create training states or candidate labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from research.refutation_bench.fc11 import source_core as original
from . import core

TRAIN_MODULES={
    "MathCan/FiniteHorizonGap.lean":(
       "epsilon_sixth_clearance","epsilon_sixth_equiv","full_horizon_recovers",
    ),
    "MathCan/DeletionMemory.lean":(
       "correct_decoder_separates","short_decks_identical","three_decks_disjoint",
    ),
    "MathCan/TOEGraph.lean":(
       "integral_add","integral_ext","integral_gradient",
       "closed_gradient_zero","closed_gauge_invariant",
       "nonzero_cycle_obstructs_potential",
    ),
    "MathCan/TOECosmology.lean":(
       "anchored_present","curvature_mode_absorbed",
       "matter_mode_absorbed","radiation_mode_absorbed",
    ),
    "MathCan/Intervention.lean":(
       "obsEqOn_refl","obsEqOn_symm",
       "obsEqOn_trans","obsEqOn_action_subset",
    ),
    "MathCan/HookState.lean":(
       "endpoint_append","pathSum_gauge",
       "closed_gauge_invariant","closed_path_nonpositive",
       "two_order_defect_gauge",
    ),
    "MathCan/GraphAuthorization.lean":(
       "repair_budget_equiv","repair_budget_mono",
    ),
}


def emit(source:Path,out:Path)->dict:
    if len(TRAIN_MODULES)!=7 or sum(len(v) for v in TRAIN_MODULES.values())!=27:
        raise AssertionError("FC12 source selection modified")
    receipt=original.emit_source(source,TRAIN_MODULES,out,phase="fc12-train-only")
    if receipt["theorems"]!=27 or receipt["stages"]<35:
        raise AssertionError("unexpected original Lean proof steps")
    receipt.update(
       experiment="FC12",
       registered_families=list(core.FAMILIES),
       train_selector_only=True,
       test_selection_exists_in_this_revision=False,
       tactic_context_arguments_only=True,
    )
    (out/"FC12_train_source_receipt.json").write_text(
         json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def extract(out:Path)->dict:
    report=original.extract_states(out)
    states=core.states_from_compiler(out)
    count=sum(r["goal_count"]==1 for r in states)
    if count<25:
        raise AssertionError("fewer than 25 available real single-goal training states")
    from collections import Counter
    avail=Counter()
    for row in states:
        if row["goal_count"]!=1:
            continue
        for name,candidates in core.generate(row["actual_lean_goal_state"]).items():
            if candidates:
                avail[name]+=1
    report["single_goal_training_states"]=count
    report["family_context_generated_count"]=dict(avail)
    (out/"FC12_train_goal_receipt.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--extract",action="store_true")
    a=p.parse_args()
    response=extract(a.out) if a.extract else emit(a.source,a.out)
    print(json.dumps(response,sort_keys=True,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
