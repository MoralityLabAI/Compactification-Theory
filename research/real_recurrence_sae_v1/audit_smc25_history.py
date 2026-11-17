"""SMC-25H: post-result source-only historical roster exclusion audit.

All generations use the same SMC-13 registered deterministic MRV Sudoku
source. No pretrained teacher weights, SAE, task gradients, interventions,
or scored numerical effects are loaded by this independent audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from smc13_family import (
    build_rosters,new_full_solution,dig_unique,valid_solution,
    rectangle_count,count_solutions,fingerprint
)

HISTORICAL={
    "smc15_novel30":(202615030000,30),
    "smc15_novel34":(202615040000,34),
    "smc16_novel30":(202616030000,30),
    "smc16_novel34":(202616040000,34),
    "smc23_novel30":(202633030000,30),
    "smc23_novel34":(202633040000,34),
    "smc24_novel30":(202634030000,30),
    "smc24_novel34":(202634040000,34),
}
SMC25={
    "smc25_novel30":(202635030000,30),
    "smc25_novel34":(202635040000,34),
}
REGISTERED_SMC25_MANIFEST={
    "smc25_novel30":"24e43d7621c8a6c2c473a1c18a3ea932886ba370d7a5d9e5c230066df2219c89",
    "smc25_novel34":"3d60f66f24d98e98ec1a80f1789fa20c289ad7ba5530c2ab9dd4c09d0b3c1ae4",
}
N=64


def regenerate(base,holes):
    cases=[]
    for j in range(N):
        sd=base+j
        solution=new_full_solution(sd)
        puzzle=dig_unique(solution,sd+1000000,holes)
        if not valid_solution(solution) or rectangle_count(solution)<=0:
            raise ValueError("noncanonical source reproducibility failure")
        if puzzle.count(0)!=holes or count_solutions(puzzle,2)!=1:
            raise ValueError("invalid unique Sudoku source puzzle")
        if any(a and a!=b for a,b in zip(puzzle,solution)):
            raise ValueError("puzzle clue contradicts registered complete solution")
        cases.append({"seed":sd,"x":puzzle,"target":solution,
                      "holes":holes,"rectangles":rectangle_count(solution)})
    return cases


def run():
    source13,source_manifest=build_rosters()
    prior_inputs={tuple(x["x"]) for cohort in source13.values() for x in cohort}
    prior_targets={tuple(x["target"]) for cohort in source13.values() for x in cohort}
    count_13=sum(len(x) for x in source13.values())
    if len(prior_inputs)!=count_13 or len(prior_targets)!=count_13 or count_13!=512:
        raise ValueError("original SMC13 source registry not full/disjoint")
    old_fingerprints={}
    old_counts={}
    for family,params in HISTORICAL.items():
        cases=regenerate(*params)
        old_fingerprints[family]=fingerprint(cases)
        old_counts[family]=len(cases)
        prior_inputs.update(tuple(r["x"]) for r in cases)
        prior_targets.update(tuple(r["target"]) for r in cases)
    if len(prior_inputs)!=1024 or len(prior_targets)!=1024:
        raise ValueError("expected exactly 1024 historical distinct inputs and targets")
    source25_inputs=set()
    source25_targets=set()
    fresh={}
    for family,params in SMC25.items():
        cases=regenerate(*params)
        observed=fingerprint(cases)
        if observed!=REGISTERED_SMC25_MANIFEST[family]:
            raise ValueError("actual prior-test SMC25 source roster SHA mismatch")
        ids={tuple(r["x"]) for r in cases}
        tars={tuple(r["target"]) for r in cases}
        if len(ids)!=N or len(tars)!=N:
            raise ValueError("SMC25 new cohort source repeats itself")
        if ids & (prior_inputs|source25_inputs) or tars & (prior_targets|source25_targets):
            raise ValueError("SMC25 test source overlaps registered historical cohorts")
        source25_inputs.update(ids)
        source25_targets.update(tars)
        fresh[family]={
            "roster_sha256":observed,"n":len(cases),
            "min_intercalates":min(r["rectangles"] for r in cases),
            "max_intercalates":max(r["rectangles"] for r in cases),
            "overlap_previous_source_inputs":0,
            "overlap_previous_source_targets":0,
        }
    if len(source25_inputs)!=128 or len(source25_targets)!=128:
        raise ValueError("SMC25 source family disjointness incomplete")
    return {
        "schema":"SMC25/post-result-source-only-historical-roster-audit-v1",
        "status":"PASS",
        "status_kind":"retrospective source provenance only; no heldout numerical effect rescoring",
        "source_generator":"frozen SMC13 randomized MRV plus unique-solution digger",
        "SMC13_source_count":count_13,
        "additional_historical_source_count":sum(old_counts.values()),
        "total_prior_source_inputs_and_targets":len(prior_inputs),
        "new_SMC25_source_count":len(source25_inputs),
        "previous_fingerprints":old_fingerprints,
        "registered_SMC25_test_rosters":fresh,
        "scientific_thresholds_changed":False,
        "raw_test_outcomes_accessed":False,
        "arg03_sealed_data_accessed":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("SMC25_HISTORICAL_SOURCE_AUDIT_PASS",result["total_prior_source_inputs_and_targets"],
          result["new_SMC25_source_count"],flush=True)


if __name__=="__main__":
    main()
