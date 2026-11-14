"""FC-11 TRAIN-ONLY original MathCan theorem trajectory manifest.

This module contains NO held-out theorem selections. It is frozen and SHA
pinned before adding test_data.py. The original MathCan checkout remains
unchanged; the generated instrumented copies compile under Lean4.35.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import source_core as core

TRAIN_SOURCE={
    "MathCan/FiniteHorizonGap.lean":(
        "epsilon_sixth_clearance",
        "epsilon_sixth_equiv",
        "full_horizon_recovers",
        "mask_period_63",
    ),
    "MathCan/TOECosmology.lean":(
        "anchored_present",
        "curvature_mode_absorbed",
        "matter_mode_absorbed",
        "radiation_mode_absorbed",
        "background_present",
        "conditional_quadratic_blind_mode",
    ),
    "MathCan/TOEGraph.lean":(
        "integral_ext",
        "closed_gradient_zero",
        "closed_gauge_invariant",
        "nonzero_cycle_obstructs_potential",
    ),
    "MathCan/DeletionMemory.lean":(
        "correct_decoder_separates",
        "short_decks_identical",
    ),
}


def emit(source:Path,out:Path)->dict:
    report=core.emit_source(source,TRAIN_SOURCE,out,phase="train-only")
    if report["theorems"]!=16 or report["stages"]<25:
        raise AssertionError("missing registered source proof stages")
    report["heldout_selection_defined_in_this_revision"]=False
    (out/"train_receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def extract(out:Path)->dict:
    report=core.extract_states(out)
    if report["real_compiler_states"]<25:
        raise AssertionError("too few actual Lean training tactic states")
    (out/"train_extraction.json").write_text(
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
