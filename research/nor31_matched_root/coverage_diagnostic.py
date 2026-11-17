"""NOR-31 explicitly RETROSPECTIVE finite-batch coordinate coverage diagnostic.

Read the committed ORIGINAL eight-file archive. No re-running attack
selection, no new held-out data, and no modification of the preregistered
source-level PASS/FAIL. This is a structural explanation for why a
root-aware attacker can often evade a known finite audit batch.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
from math import ceil,log
from pathlib import Path
import json
import statistics
import zipfile

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/"results"/"nor31_original_hosted.zip"
RECEIPT=HERE/"results"/"nor31_first_original_receipt.json"
ARMS=("hybrid_50","uniform16","static_prior")


def ideal_unseen_mean(m,arm,n=512):
    if arm=="hybrid_50":
        N,u=m-8,8
    elif arm=="uniform16":
        N,u=m,16
    elif arm=="static_prior":
        return float(m-16)
    else:
        raise ValueError("unregistered arm")
    return N*((N-u)/N)**n


def audit():
    receipt=json.loads(RECEIPT.read_bytes())
    assert ARCHIVE.exists()
    with zipfile.ZipFile(ARCHIVE) as z:
        samples={}
        for key in ("evaluation","outcome"):
            info=receipt["original_raw"][key]
            raw=z.read(info["name"])
            assert len(raw)==info["bytes"]
            assert sha256(raw).hexdigest()==info["sha256"]
            samples[key]=json.loads(raw)
    evaluation=samples["evaluation"]
    outcome=samples["outcome"]
    assert len(evaluation["records"])==len(outcome["source_records"])==24
    groups=defaultdict(list)
    zero_hybrid_reconciled=0
    for source,graded in zip(evaluation["records"],outcome["source_records"]):
        assert source["source_sha256"]==graded["source_sha256"]
        m=source["m"]
        assert m==graded["m"] and m in (769,1537)
        for arm in ARMS:
            plans=source["schedules"][arm]
            assert len(plans)==512
            seen={i for plan in plans for i in plan}
            unseen=m-len(seen)
            leaked_rate=graded["detection"][arm]["leaked_evaluation_root"]
            # Registered greedy objective selects unseen coordinates while
            # at least four exist; if fewer than four exist, no radius-four
            # set can achieve zero batch detections.
            assert (leaked_rate==0)==(unseen>=4),(m,arm,unseen,leaked_rate)
            groups[(m,arm)].append(unseen)
            if arm=="hybrid_50":zero_hybrid_reconciled+=1
    assert zero_hybrid_reconciled==24
    result={}
    for m in (769,1537):
        result[str(m)]={}
        for arm in ARMS:
            vals=groups[(m,arm)]
            assert len(vals)==12
            result[str(m)][arm]={
                "source_count":12,
                "uninspected_mean":statistics.mean(vals),
                "uninspected_min":min(vals),
                "uninspected_max":max(vals),
                "sources_with_at_least_four_uninspected":sum(v>=4 for v in vals),
                "ideal_sampler_expected_uninspected":ideal_unseen_mean(m,arm),
            }
    assert result["769"]["hybrid_50"]["uninspected_mean"]==41/12
    assert result["1537"]["hybrid_50"]["uninspected_mean"]==1255/12
    assert result["769"]["hybrid_50"]["sources_with_at_least_four_uninspected"]==6
    assert result["1537"]["hybrid_50"]["sources_with_at_least_four_uninspected"]==12
    assert result["769"]["uniform16"]["sources_with_at_least_four_uninspected"]==0
    assert result["1537"]["uniform16"]["sources_with_at_least_four_uninspected"]==9
    # A cohort having >=4 never-checked coordinates is sufficient for a
    # root-aware four-coordinate attack with zero detection, provided
    # it can choose those coordinates after the schedules are revealed.
    thresholds={}
    for m in (769,1537):
        N,u=m-8,8
        thresholds[str(m)]={
            "batch_count_for_ideal_E_uninspected_below_four":
                ceil(log(4/N)/log(1-u/N)),
            "batch_count_for_Markov_P_four_uninspected_at_most_0_05":
                ceil(log(0.2/N)/log(1-u/N)),
        }
    assert thresholds["769"]["batch_count_for_ideal_E_uninspected_below_four"]==497
    assert thresholds["1537"]["batch_count_for_ideal_E_uninspected_below_four"]==1134
    assert thresholds["769"]["batch_count_for_Markov_P_four_uninspected_at_most_0_05"]==781
    assert thresholds["1537"]["batch_count_for_Markov_P_four_uninspected_at_most_0_05"]==1705
    print("NOR31 RETROSPECTIVE ORIGINAL-SCHEDULE COVERAGE DIAGNOSTIC PASS")
    print(json.dumps({"coverage_by_source_length":result,
                      "illustrative_ideal_uniform_batch_scaling":thresholds,
                      "status":"retrospective explanatory analysis, not preregistered criterion"},
                     indent=2,sort_keys=True))
    return result,thresholds


if __name__=="__main__":
    audit()
