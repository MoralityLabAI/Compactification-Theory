"""Independent NOR-15C finite-policy receipt verifier.

Does not import the NOR-15C oracle implementation. It re-enumerates each
possible fixed top-four order and independently reconstructs first-positive
stopping statistics directly from the public/future membership bit masks.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

SOURCE="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1a9549a2c6"
EXPECTED="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
COHORTS=(
    "3cf6a5848d561c864d43293f98c218cd20e1d8a20f7a12b4250667b9a16444af",
    "590c68d2025504de392b02dc596afad047ac6dfd916b93c62b68509f94dfc7c8",
    "a22fcd61779b67eaa1479a7ac8f7053ead94b19d7a969da37cb6fa2c95a38692",
    "3e2b9da3bce70039d62d9f988962ca4e4fc128c4445ece642a16d6f1b2bee3f0")


def independently_score(order,rows):
    found=correct=false=checks=0
    for entry in rows:
        positive=entry["public_accepted_bitmask"]
        future=entry["future_correct_bitmask_hindsight"]
        for j in order:
            checks+=1
            if positive & (1<<j):
                found+=1
                if future & (1<<j):correct+=1
                else:false+=1
                break
    assert found==correct+false
    return {"future_correct":correct,"false_accept":false,
            "accepted":found,"paid_checks":checks,"cases":len(rows)}


def independent_audit(obj):
    assert obj["schema"]=="nor15c.retrospective_four_check_fixed_order_oracle.v1"
    assert obj["source_checkpoint_sha256"]==EXPECTED
    assert obj["nor15b_four_fresh_test_manifests"]==list(COHORTS)
    assert obj["source_case_count"]==40
    assert obj["unique_visible_source_cases"]==40
    assert obj["unique_model_visible_feature_groups"]==1
    rows=obj["case_receipts"]
    assert len(rows)==40
    assert len({x["public_visible_sha256"] for x in rows})==40
    assert len({x["public_features_sha256"] for x in rows})==1
    assert {x["replicate"] for x in rows}==set(range(4))
    assert Counter(x["replicate"] for x in rows)=={0:10,1:10,2:10,3:10}
    assert obj["public_template_check_cost_retrospective"]==640
    assert obj["future_grade_calls_retrospective"]==sum(
        x["public_accepted_bitmask"].bit_count() for x in rows)
    assert obj["unique_exact_public_acceptance_masks"]==len({
        x["public_accepted_bitmask"] for x in rows})
    assert obj["cases_with_multiple_public_accepted_templates"]==sum(
        x["public_accepted_bitmask"].bit_count()>1 for x in rows)
    assert obj["cases_with_no_future_correct_candidate_of_16"]==sum(
        x["future_correct_bitmask_hindsight"]==0 for x in rows)
    assert obj["cases_with_at_least_one_future_correct_candidate"]==sum(
        x["future_correct_bitmask_hindsight"]>0 for x in rows)
    for x in rows:
        a=x["public_accepted_bitmask"]
        c=x["future_correct_bitmask_hindsight"]
        assert 0<a<65536 and 0<=c<65536 and c&~a==0
        assert x["accepted_template_count"]==a.bit_count()
        assert x["future_correct_template_count"]==c.bit_count()
        assert x["has_a_publicly_ambiguous_accept_set"]==(a.bit_count()>1)
    sigs=Counter(x["public_accepted_bitmask"] for x in rows)
    histogram=Counter(sigs.values())
    assert {str(k):v for k,v in sorted(histogram.items())}==obj[
        "public_acceptance_group_size_histogram"]
    rank=obj["original_frozen_model_top_four"]
    assert len(set(rank))==4 and all(0<=j<16 for j in rank)
    assert independently_score(rank,rows)==obj["original_fixed_four_outcome"]
    assert obj["original_fixed_four_outcome"]["future_correct"]==8
    assert obj["original_fixed_four_outcome"]["false_accept"]==20
    oracle=obj["retrospective_hindsight_fixed_order_oracle"]
    assert oracle["candidate_count"]==16 and oracle["check_budget"]==4
    assert oracle["distinct_fixed_orders_exhaustively_checked"]==43680
    assert oracle["posterior_test_future_used_to_select_oracle"]
    assert not oracle["inference_feasible_untrained_oracle"]
    assert obj["scope"]["best_order_uses_test_future_labels_and_is_non_deployable"]
    assert obj["scope"]["retrospective_public_checks_not_charged_to_native_inference"]
    assert obj["scope"]["no_adaptive_stop_after_positive"]

    winner=None
    win_metric=None
    accuracy_false_pairs=set()
    mins={}
    best_checks={}
    for order in permutations(range(16),4):
        score=independently_score(order,rows)
        key=(score["future_correct"],score["false_accept"])
        accuracy_false_pairs.add(key)
        if score["future_correct"] not in mins or score["false_accept"]<mins[score["future_correct"]]:
            mins[score["future_correct"]]=score["false_accept"]
        if key not in best_checks or score["paid_checks"]<best_checks[key]:
            best_checks[key]=score["paid_checks"]
        desc=(score["future_correct"],-score["false_accept"],-score["paid_checks"])
        if win_metric is None or desc > (win_metric["future_correct"],
                                          -win_metric["false_accept"],
                                          -win_metric["paid_checks"]):
            winner=list(order)
            win_metric=score
    assert oracle["hindsight_optimal_order"]==winner
    assert oracle["hindsight_optimal_metrics"]==win_metric
    assert oracle["distinct_accuracy_false_accept_pairs"]==len(accuracy_false_pairs)
    assert oracle["future_correct_values_attainable"]==sorted(mins)
    actual_frontier=[{"future_correct":k,
                      "min_false_accept":mins[k],
                      "min_paid_checks_at_this_accuracy_and_false_accept":
                          best_checks[(k,mins[k])]}
                     for k in sorted(mins)]
    assert actual_frontier==oracle["min_false_accept_for_each_future_correct"]
    return {"alias_cases":40,"fixed_orderings_checked":43680,
            "hindsight_future_correct_ceiling":win_metric["future_correct"],
            "hindsight_false_accept_at_ceiling":win_metric["false_accept"],
            "public_checks_charged_for_audit":640,
            "model_only_original_future_correct":8}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",type=Path,required=True)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    raw=args.full.read_bytes()
    obj=json.loads(raw)
    result=independent_audit(obj)
    print("NOR15C INDEPENDENT EXACT FOUR-CHECK ORDER AUDIT PASSED",result)
    print("NOR15C JSON",len(raw),sha256(raw).hexdigest())
    if args.check:
        assert result["fixed_orderings_checked"]==43680


if __name__=="__main__":
    main()
