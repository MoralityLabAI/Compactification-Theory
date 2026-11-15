"""Independent full NOR-15C public mask and future-oracle receipt audit.

Does not import source data-generation or experiment code. Recomputes
all 43,680 ordered four-check policies from frozen case bit masks and
verifies exact finite upper bounds. Hindsight future-label optimisation
is diagnostic only and cannot be deployed as public-only inference.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor15c_frozen_receipt.json"
SHA="11b90206703a3e21c7163435de9f95d6c212cb320bf0a663bc86d0ce4981236e"
CK="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def frozen():
    z=json.loads(RECEIPT.read_text())
    assert z["schema"]=="nor15c.first_independent_feature_fibre_oracle_receipt.v1"
    assert z["full_json_sha256"]==SHA and z["full_json_bytes"]==205705
    assert z["checkpoint_sha256"]==CK
    assert z["restricted_policy_orders_exhaustively_evaluated"]==43680
    assert z["dev_source_cases"]==80 and z["test_source_cases"]==160
    assert z["shared_model_input_groups"]==1
    assert z["test_original_score"]==[160,30,77,53,387,107]
    assert z["test_hindsight_best_score"]==[160,49,79,32,368,128]
    assert z["full_mask_per_pattern_hindsight_correct"]==84
    assert not z["population_bound_claimed"]
    return z


def one(mask,truth,seq):
    for index,j in enumerate(seq,1):
        if (mask>>j)&1:
            return (int(bool((truth>>j)&1)),
                    int(not ((truth>>j)&1)),
                    0,index,1)
    return (0,0,1,len(seq),0)


def score(cases,order):
    result=[0,0,0,0,0]
    counts=Counter((r["public_accepted_mask_16"],
                    r["withheld_future_correct_mask_16"]) for r in cases)
    for (mask,correct),weight in counts.items():
        partial=one(mask,correct,order)
        for k in range(5):
            result[k]+=weight*partial[k]
    success,false,abstain,checks,verified=result
    assert success+false+abstain==len(cases)
    return {"cases":len(cases),"future_correct":success,"false_accept":false,
            "abstained":abstain,"paid_checks":checks,"verified":verified}


def best(cases):
    groups=Counter((r["public_accepted_mask_16"],
                    r["withheld_future_correct_mask_16"]) for r in cases)
    optimum=None;best_sequence=None;best_metrics=None
    for order in permutations(range(16),4):
        success=false=checks=0
        for (mask,truth),n in groups.items():
            s,f,ab,c,v=one(mask,truth,order)
            success+=n*s;false+=n*f;checks+=n*c
        key=(success,-false,-checks,tuple(-i for i in order))
        if optimum is None or key>optimum:
            optimum=key;best_sequence=order
            best_metrics={"future_correct":success,"false_accept":false,
                          "paid_checks":checks}
    return best_sequence,best_metrics


def mask_ceiling(cases):
    groups=defaultdict(list)
    for row in cases:groups[row["public_accepted_mask_16"]].append(row)
    best_total=0;details={}
    for mask,items in groups.items():
        counts={j:sum(bool((r["withheld_future_correct_mask_16"]>>j)&1)
                      for r in items) for j in range(16) if (mask>>j)&1}
        if counts:
            top=max(counts,key=lambda j:(counts[j],-j))
            best=counts[top]
        else:
            top=None;best=0
        best_total+=best
        details[mask]=(len(items),top,best)
    future_available=sum(bool(row["withheld_future_correct_mask_16"]) for row in cases)
    return details,best_total,future_available


def verify_full(data,z):
    assert data["schema"]=="nor15c.feature_fibre_exact_four_check_hindsight_ceiling.v1"
    assert data["source_checkpoint_sha256"]==CK
    assert data["frozen_probe_sha256"]==z["frozen_probe_sha256"]
    assert data["train_manifest"]==z["train_manifest"]
    assert data["dev_cases"]==80 and data["test_cases"]==160
    assert data["distinct_public_visible_sources"]==240
    assert data["unique_model_visible_feature_hashes"]==1
    assert data["common_model_visible_feature_sha256"]==z["shared_model_input_digest"]
    assert data["dev_source_manifest_sha256"]==z["dev_visible_manifest"]
    assert data["test_source_manifest_sha256"]==z["test_visible_manifest"]
    assert data["separate_diagnostic_public_16_template_checks"]==3840
    assert data["separate_diagnostic_withheld_future_16_grades"]==3840
    scope=data["scope"]
    assert scope["nonadaptive_shared_feature_order_class_only"]
    assert scope["all_oracle_future_labels_scoring_or_dev_diagnostic_only"]
    assert scope["test_future_labels_absent_from_dev_selection"]
    assert scope["hindsight_test_oracle_is_nondeployable"]
    assert scope["does_not_certify_population_ceiling"]
    dev=data["dev_case_records"];test=data["test_case_records"]
    assert len(dev)==80 and len(test)==160
    assert len({r["public_visible_sha256"] for r in dev+test})==240
    assert len({r["public_features_sha256"] for r in dev+test})==1
    assert {r["public_features_sha256"] for r in dev+test}=={z["shared_model_input_digest"]}
    assert sha256("\n".join(r["public_visible_sha256"] for r in dev).encode()).hexdigest()==z["dev_visible_manifest"]
    assert sha256("\n".join(r["public_visible_sha256"] for r in test).encode()).hexdigest()==z["test_visible_manifest"]
    for r in dev+test:
        assert r["source_family"]=="late_source"
        accepted=r["public_accepted_mask_16"]
        correct=r["withheld_future_correct_mask_16"]
        assert 0<=accepted<2**16 and 0<=correct<2**16
        assert correct & ~accepted==0
        assert r["public_checks_for_diagnostic_all_16"]==16
        assert r["future_grades_diagnostic_all_16"]==16
        assert len(r["original_order_16"])==16
        assert sorted(r["original_order_16"])==list(range(16))
        assert r["original_top4"]==z["original_top4"]
    assert data["original_shared_four_order"]==z["original_top4"]
    orig=score(test,z["original_top4"])
    assert [orig[k] for k in z["score_fields"]]==z["test_original_score"]
    assert orig==data["test_original_shared_model_policy"]
    dseq,dmetrics=best(dev)
    assert list(dseq)==z["dev_hindsight_top4"]
    devscore=score(dev,dseq)
    assert [devscore[k] for k in z["score_fields"]]==z["dev_hindsight_score"]
    recordeddev=data["development_future_selected_four_order"]
    assert recordeddev["orders_exhaustively_checked"]==43680
    assert recordeddev["ordered_top4"]==list(dseq)
    for key in devscore:assert recordeddev[key]==devscore[key]
    selected=score(test,dseq)
    assert [selected[k] for k in z["score_fields"]]==z["test_development_chosen_score"]
    assert selected==data["test_dev_selected_diagnostic_policy"]
    tseq,tmetrics=best(test)
    assert list(tseq)==z["test_hindsight_top4"]
    tscore=score(test,tseq)
    assert [tscore[k] for k in z["score_fields"]]==z["test_hindsight_best_score"]
    recordedtest=data["test_oracle_hindsight_optimal_four_order"]
    assert recordedtest["orders_exhaustively_checked"]==43680
    assert recordedtest["ordered_top4"]==list(tseq)
    for key in tscore:assert recordedtest[key]==tscore[key]
    rand=data["test_seeded_random_shared_policy"]
    randscore=score(test,rand["order"])
    for key in randscore:assert rand[key]==randscore[key]
    patterns,total,percase=mask_ceiling(test)
    recordedmask=data["test_full_mask_group_hindsight_bounds"]
    assert len(patterns)==z["public_mask_equivalence_groups"]==26
    assert total==z["full_mask_per_pattern_hindsight_correct"]==84
    assert percase==z["per_case_future_clairvoyant_correct"]==160
    for pattern in recordedmask["groups"]:
        mask=pattern["public_accepted_mask_16"]
        assert patterns[mask]==(
            pattern["cases"],pattern["hindsight_best_candidate"],
            pattern["hindsight_future_correct"])
    assert patterns[65535][0]==z["fully_ambiguous_public_mask_65535_cases"]==41
    assert patterns[65535][2]==z["fully_ambiguous_public_mask_65535_best_pattern_correct"]==6
    assert tscore["future_correct"]<=total<=percase
    return {"source_cases":240,"shared_model_inputs":1,
            "exact_four_check_orders_reenumerated_on_each_cohort":43680,
            "public_acceptance_pattern_groups":26,
            "hindsight_four_check_ceiling":49,
            "mask_conditioned_hindsight_ceiling":84,
            "per_case_future_oracle_ceiling":160}


def main():
    p=argparse.ArgumentParser()
    inputs=p.add_mutually_exclusive_group()
    inputs.add_argument("--full",type=Path)
    inputs.add_argument("--archive",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    receipt=frozen()
    raw=None
    if args.full:raw=args.full.read_bytes()
    if args.archive:
        with ZipFile(args.archive) as archive:
            assert archive.namelist()==["nor15c_full.json"]
            assert archive.testzip() is None
            raw=archive.read("nor15c_full.json")
    if raw is not None:
        assert len(raw)==receipt["full_json_bytes"]
        assert sha256(raw).hexdigest()==SHA
        checked=verify_full(json.loads(raw),receipt)
        print("NOR15C INDEPENDENT ENUMERATED ORACLE AUDIT PASSED",checked)
    else:
        print("NOR15C FROZEN FIRST FULL RECEIPT VALID")
    if args.check:assert receipt["first_hosted_ci"].startswith("https://github.com/")


if __name__=="__main__":
    main()
