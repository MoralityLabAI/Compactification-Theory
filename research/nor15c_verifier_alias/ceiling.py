"""NOR-15C: exact fixed-four verifier order ceiling on aliased public inputs.

Forty NOR-15B held-out late_source examples share one 9x8 public-feature
tensor. Every deterministic representation-only ranker must therefore
send the same candidate ordering to the four-check exact public verifier.
We enumerate ALL 16P4 = 43,680 ordered four-candidate policies.

This is strictly a RETROSPECTIVE, hindsight upper-bound audit.
Sixteen exact public checks per case and withheld future grading per
public-accepted class are charged separately from native four-check
inference. Test future correctness is used ONLY to measure the oracle
ceiling, never to train, tune, or select a deployable policy.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import bench25, compound25 as corpus, model25
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as nor12
from research.nor13_gated_readout import interaction as nor13
from research.nor15_trajectory_sufficiency import task_test as nor15b

SCHEMA="nor15c.retrospective_four_check_fixed_order_oracle.v1"
EXPECTED_CHECKPOINT="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
EXPECTED_NOR15B_TEST_MANIFESTS=(
    "3cf6a5848d561c864d43293f98c218cd20e1d8a20f7a12b4250667b9a16444af",
    "590c68d2025504de392b02dc596afad047ac6dfd916b93c62b68509f94dfc7c8",
    "a22fcd61779b67eaa1479a7ac8f7053ead94b19d7a969da37cb6fa2c95a38692",
    "3e2b9da3bce70039d62d9f988962ca4e4fc128c4445ece642a16d6f1b2bee3f0")
CANDIDATES=16
BUDGET=4


def masks_from_one_case(case,header):
    """Exact test-public candidate checks, separately charged as hindsight."""
    accepted=0
    for j in range(CANDIDATES):
        if corpus.verify_template(case,header,corpus.OPS[j//4],corpus.OPS[j%4]):
            accepted |= 1<<j
    assert accepted, "each generated positive source has a public-accepted template"
    correct=0
    for j in range(CANDIDATES):
        if not accepted & (1<<j):
            continue
        program=(header.base,header.delay,header.spacing,header.amplitude,
                 corpus.OPS[j//4],corpus.OPS[j%4])
        # This call accesses the withheld FUTURE and is restricted to
        # the retrospectively evaluated, after-the-fact oracle bound.
        grade=corpus.grade(case,program)
        assert grade["accepted"]==1
        if grade["future_correct"]:
            correct |= 1<<j
    assert correct & ~accepted==0
    return accepted,correct


def score_order_masks(accepted_masks,correct_masks,order):
    """Return three mutually consistent metrics for a fixed top-k order."""
    if len(accepted_masks)!=len(correct_masks):
        raise ValueError("mismatched source masks")
    if len(order)!=len(set(order)) or not all(
        isinstance(j,int) and 0<=j<CANDIDATES for j in order):
        raise ValueError("candidate indices must be distinct and in range")
    correct=false_accept=paid=accepted=0
    for A,C in zip(accepted_masks,correct_masks,strict=True):
        if C & ~A:raise AssertionError("future-positive set exceeds public accepted set")
        found=False
        for j in order:
            paid+=1
            if A & (1<<j):
                accepted+=1
                correct+=bool(C & (1<<j))
                false_accept+=not bool(C & (1<<j))
                found=True
                break
        if not found:
            assert (paid<=len(order)*len(accepted_masks))
    assert correct+false_accept==accepted
    return {"future_correct":correct,"false_accept":false_accept,
            "accepted":accepted,"paid_checks":paid,
            "cases":len(accepted_masks)}


def enumerate_fixed_rankings(accepted_masks,correct_masks,candidates=CANDIDATES,
                             budget=BUDGET):
    """Exact 16P4 exhaustive oracle; future labels optimize hindsight only."""
    if candidates<budget or budget<1 or candidates>16:
        raise ValueError("bounded finite policy enumeration required")
    if len(accepted_masks)!=len(correct_masks) or not accepted_masks:
        raise ValueError("need matched positive finite source masks")
    if any(C&~A for A,C in zip(accepted_masks,correct_masks,strict=True)):
        raise ValueError("incorrect ground-truth oracle mask")
    # Enumerate deterministic ordered sequences of four candidate checks.
    # With only negative feedback before a first successful verification,
    # any deterministic, feature-only, stop-at-first-positive adaptive
    # policy follows one preassigned sequence of tests for this feature.
    max_correct=-1;winner=None;best_metrics=None
    counts=Counter()
    min_false_given_correct={}
    min_checks_given_correct_and_false={}
    n=0
    for order in permutations(range(candidates),budget):
        metrics=score_order_masks(accepted_masks,correct_masks,order)
        n+=1
        c=metrics["future_correct"]
        f=metrics["false_accept"]
        paid=metrics["paid_checks"]
        counts[(c,f)]+=1
        min_false_given_correct[c]=min(f,min_false_given_correct.get(c,f))
        key=(c,f)
        min_checks_given_correct_and_false[key]=min(
            paid,min_checks_given_correct_and_false.get(key,paid))
        candidate=(c,-f,-paid)
        if (winner is None or candidate > (
            best_metrics["future_correct"],
            -best_metrics["false_accept"],-best_metrics["paid_checks"])):
            winner=order
            best_metrics=metrics
    assert n==len(list(permutations(range(candidates),budget)))
    assert best_metrics["future_correct"]==max(min_false_given_correct)
    assert best_metrics["false_accept"]==min_false_given_correct[
        best_metrics["future_correct"]]
    return {
        "candidate_count":candidates,
        "check_budget":budget,
        "distinct_fixed_orders_exhaustively_checked":n,
        "hindsight_optimal_order":list(winner),
        "hindsight_optimal_metrics":best_metrics,
        "future_correct_values_attainable":sorted(min_false_given_correct),
        "min_false_accept_for_each_future_correct":[{
            "future_correct":c,
            "min_false_accept":min_false_given_correct[c],
            "min_paid_checks_at_this_accuracy_and_false_accept":
                min_checks_given_correct_and_false[(c,min_false_given_correct[c])],
        } for c in sorted(min_false_given_correct)],
        "distinct_accuracy_false_accept_pairs":len(counts),
        "posterior_test_future_used_to_select_oracle":True,
        "inference_feasible_untrained_oracle":False,
    }


def original_order(case,header,checkpoint):
    scores=model25.logits(case,header,checkpoint)
    return sorted(range(CANDIDATES),key=lambda j:(-scores[j],j))[:BUDGET]


def run():
    # Reconstruct only the public original train/dev/test roster, skipping
    # any fitted NOR-12 through 15 model readouts.
    roster=nor11.sample_cohorts(nor12.SPLIT_COUNTS)
    assert sha256("\n".join(d for _,_,d in roster["train"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["train"]
    assert sha256("\n".join(d for _,_,d in roster["dev"]).encode()
                  ).hexdigest()==nor12.BASELINE_MANIFESTS["dev"]
    cohorts=nor15b.source_disjoint_new(roster)
    assert len(cohorts)==4 and all(len(batch)==260 for batch in cohorts)
    manifests=tuple(sha256("\n".join(d for _,_,d in batch).encode()
                           ).hexdigest() for batch in cohorts)
    assert manifests==EXPECTED_NOR15B_TEST_MANIFESTS
    late=[(case,header,digest,rep)
          for rep,batch in enumerate(cohorts)
          for case,header,digest in batch if case.family=="late_source"]
    assert len(late)==40 and len({digest for _,_,digest,_ in late})==1
    assert len({nor13.source_fingerprint(case) for case,_,_,_ in late})==40

    ck,digests=bench25.checkpoints()
    frozen=ck["recurrent"]
    assert frozen["parameters"]==1000 and digests["recurrent"]==EXPECTED_CHECKPOINT
    rank=original_order(late[0][0],late[0][1],frozen)
    original_logits_hash=sha256(json.dumps(
        model25.logits(late[0][0],late[0][1],frozen),
        separators=(",",":")).encode()).hexdigest()
    A=[];C=[];sources=[]
    for case,header,digest,rep in late:
        # Same model-visible tensor means identical deterministic original
        # logits. This does not imply identical exact verifier responses.
        assert original_order(case,header,frozen)==rank
        a,c=masks_from_one_case(case,header)
        A.append(a);C.append(c)
        sources.append({"public_features_sha256":digest,
                        "public_visible_sha256":nor13.source_fingerprint(case),
                        "replicate":rep,
                        "public_accepted_bitmask":a,
                        "future_correct_bitmask_hindsight":c,
                        "accepted_template_count":a.bit_count(),
                        "future_correct_template_count":c.bit_count(),
                        "has_a_publicly_ambiguous_accept_set":a.bit_count()>1})
    baseline=score_order_masks(A,C,rank)
    # Original frozen 1000-param GRU score on the same NOR-15B 40-case
    # late_source cohort must agree with the published immutable receipt.
    assert baseline["cases"]==40
    assert baseline["future_correct"]==8
    assert baseline["false_accept"]==20
    oracle=enumerate_fixed_rankings(A,C)
    groups=defaultdict(list)
    for idx,row in enumerate(sources):
        groups[row["public_accepted_bitmask"]].append(idx)
    identical_acceptance_signature_counts=Counter(map(len,groups.values()))
    return {
        "schema":SCHEMA,
        "source_checkpoint_sha256":EXPECTED_CHECKPOINT,
        "source_nonlate_train_manifest":nor12.BASELINE_MANIFESTS["train"],
        "source_dev_manifest":nor12.BASELINE_MANIFESTS["dev"],
        "nor15b_four_fresh_test_manifests":list(manifests),
        "source_case_count":40,
        "unique_visible_source_cases":40,
        "unique_model_visible_feature_groups":1,
        "feature_digest":late[0][2],
        "original_frozen_model_top_four":rank,
        "original_frozen_model_logits_sha256":original_logits_hash,
        "original_fixed_four_outcome":baseline,
        "retrospective_hindsight_fixed_order_oracle":oracle,
        "public_template_check_cost_retrospective":CANDIDATES*len(late),
        "future_grade_calls_retrospective":sum(a.bit_count() for a in A),
        "unique_exact_public_acceptance_masks":len(groups),
        "public_acceptance_group_size_histogram":dict(sorted(
            identical_acceptance_signature_counts.items())),
        "cases_with_multiple_public_accepted_templates":sum(
            a.bit_count()>1 for a in A),
        "cases_with_no_future_correct_candidate_of_16":sum(c==0 for c in C),
        "cases_with_at_least_one_future_correct_candidate":sum(c>0 for c in C),
        "case_receipts":sources,
        "scope":{
            "hindsight_future_is_never_training_input":True,
            "retrospective_public_checks_not_charged_to_native_inference":True,
            "best_order_uses_test_future_labels_and_is_non_deployable":True,
            "exact_over_all_fixed_four_orderings":True,
            "no_adaptive_stop_after_positive":True,
            "source_family_late_source_aliased_input_only":True,
            "future_grade_and_accepted_sets_disclosed_for_audit":True,
            "native_original_four_verification_budget_unchanged":True,
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=run()
    if args.check:
        assert result["source_case_count"]==40
        assert result["unique_model_visible_feature_groups"]==1
        assert result["retrospective_hindsight_fixed_order_oracle"][
            "distinct_fixed_orders_exhaustively_checked"]==43680
        assert result["original_fixed_four_outcome"]["future_correct"]==8
        assert result["scope"]["best_order_uses_test_future_labels_and_is_non_deployable"]
        assert result["retrospective_hindsight_fixed_order_oracle"][
            "hindsight_optimal_metrics"]["future_correct"]>=8
    doc=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(doc)
    else:
        print(doc,end="")


if __name__=="__main__":
    main()
