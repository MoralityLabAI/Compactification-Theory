"""NOR-15C: exhaustive four-check fixed-ranking ceiling in one model-feature fibre.

All 16 exact PUBLIC acceptance bits are independently verified for each
case. The 16 future correctness bits require WITHHELD degree-256 grading,
and are used only for diagnostic development policy selection, final
scoring and posthoc upper bounds. A ranker using only the single shared
nine-by-eight feature tensor produces one candidate order across all cases.

The max across all 16P4=43,680 ordered four-candidate policies on TEST
is a hindsight finite-sample oracle bound, NOT a deployable policy or
population/conditional-mutual-information theorem.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus,model25
from research.nor12_task_readout import readout as old
from research.nor12_task_readout import independent_confirmation as oldfresh
from research.nor13_gated_readout import interaction as nor13
from research.nor14_conditional_basin import experiment as nor14
from research.nor15_trajectory_sufficiency import audit as nor15a
from research.nor15_trajectory_sufficiency import task_test as nor15b

SCHEMA="nor15c.feature_fibre_exact_four_check_hindsight_ceiling.v1"
DEV_OFFSET=93015000
TEST_OFFSET=93115000
DEV_CASES=80
TEST_CASES=160
PUBLIC_TEMPLATE_COUNT=16
PAID_CHECK_BUDGET=4
FAMILY="late_source"
TOTAL_ORDERS=16*15*14*13
RANDOM_ORDER_SEED=202611151513


def visible_hash(case):
    return nor13.source_fingerprint(case)


def all_prior_source_fingerprints(roster):
    seen={visible_hash(case) for split in roster.values() for case,head,d in split}
    previous=[
        oldfresh.fresh_cohorts(roster,replicates=oldfresh.REPETITIONS,
                              per_family=oldfresh.CASES_PER_FAMILY),
        nor13.new_heldout(roster,replicates=nor13.CONFIRMATIONS,
                          per_family=nor13.FAMILY_PER),
        nor14.fresh_cohorts(roster,repetitions=nor14.REPETITIONS,
                            per_family=nor14.PER_FAMILY),
        nor15a.new_heldout(roster,repetitions=nor15a.REPETITIONS,
                           per_family=nor15a.PER_FAMILY),
        nor15b.source_disjoint_new(roster,reps=nor15b.REPS,
                                  nonlate=nor15b.NONLATE_PER,
                                  late=nor15b.LATE_PER)]
    for batches in previous:
        for cohort in batches:
            for case,header,digest in cohort:
                seen.add(visible_hash(case))
    return seen


def late_source_new(seen,n,offset,label):
    if not 1<=n<=160 or offset not in (DEV_OFFSET,TEST_OFFSET):
        raise ValueError("frozen NOR15C source budget exceeded")
    rng=random.Random(corpus.SEED+offset)
    result=[]
    for trial in range(n*4000):
        case=corpus.make_case(rng,FAMILY,f"nor15c-{label}-{trial:06d}")
        headers=corpus.possible_headers(case)
        if len(headers)!=1:continue
        vis=visible_hash(case)
        if vis in seen:continue
        seen.add(vis)
        head=headers[0]
        digest=nor14.prior_probe.public_digest(case,head)
        result.append((case,head,digest,vis))
        if len(result)==n:break
    if len(result)!=n:raise RuntimeError("late_source public case exhaustion")
    return result


def observe_one(item,ck):
    case,header,digest,vis=item
    hidden,parity=nor14.prior_probe.model_prefix_hidden(case,header,ck)
    assert parity<1e-11
    scores=model25.affine(ck["state_dict"]["head.weight"],
                          ck["state_dict"]["head.bias"],hidden)
    original_order=sorted(range(PUBLIC_TEMPLATE_COUNT),
                          key=lambda j:(-scores[j],j))
    accepted=correct=0
    # Exact public demo verification and evaluation-only withheld future
    # grading are computed separately and charged as different oracles.
    for j in range(PUBLIC_TEMPLATE_COUNT):
        u=corpus.OPS[j//4]
        v=corpus.OPS[j%4]
        if corpus.verify_template(case,header,u,v):
            accepted|=1<<j
        candidate=(header.base,header.delay,header.spacing,
                   header.amplitude,u,v)
        graded=corpus.grade(case,candidate)
        if graded["future_correct"]:
            assert graded["accepted"] and (accepted>>j)&1
            correct|=1<<j
    assert correct & ~accepted==0
    return {
        "public_features_sha256":digest,
        "public_visible_sha256":vis,
        "public_accepted_mask_16":accepted,
        "withheld_future_correct_mask_16":correct,
        "original_order_16":original_order,
        "original_top4":original_order[:4],
        "hidden_state_fingerprint":sha256(json.dumps(
            hidden,separators=(",",":")).encode()).hexdigest(),
        "public_checks_for_diagnostic_all_16":16,
        "future_grades_diagnostic_all_16":16,
        "source_family":case.family,
    }


def rank_one(accepted,correct,order):
    """Outcome of first exact-accepted candidate under a bounded fixed order."""
    assert len(order)==PAID_CHECK_BUDGET and len(set(order))==PAID_CHECK_BUDGET
    assert all(0<=j<PUBLIC_TEMPLATE_COUNT for j in order)
    for n,j in enumerate(order,1):
        if accepted & (1<<j):
            return {"future_correct":int(bool(correct & (1<<j))),
                    "false_accept":int(not (correct & (1<<j))),
                    "verified":1,"abstained":0,"paid_checks":n,
                    "accepted_template":j}
    return {"future_correct":0,"false_accept":0,"verified":0,
            "abstained":1,"paid_checks":PAID_CHECK_BUDGET,
            "accepted_template":None}


def score_order(cases,order):
    tally=Counter()
    for row in cases:
        r=rank_one(row["public_accepted_mask_16"],
                   row["withheld_future_correct_mask_16"],order)
        for k in ("future_correct","false_accept","verified","abstained","paid_checks"):
            tally[k]+=r[k]
    assert tally["verified"]+tally["abstained"]==len(cases)
    assert tally["verified"]==tally["future_correct"]+tally["false_accept"]
    assert tally["paid_checks"]<=PAID_CHECK_BUDGET*len(cases)
    return {"cases":len(cases),**dict(tally)}


def score_unique_signatures(cases,order):
    grouped=Counter((r["public_accepted_mask_16"],
                     r["withheld_future_correct_mask_16"]) for r in cases)
    correct=false=checks=0
    for (acc,future),count in grouped.items():
        outcome=rank_one(acc,future,order)
        correct+=count*outcome["future_correct"]
        false+=count*outcome["false_accept"]
        checks+=count*outcome["paid_checks"]
    return {"future_correct":correct,"false_accept":false,
            "paid_checks":checks}


def optimize_order(cases):
    """An exact exhaustive oracle over all distinct ordered 4-tuples."""
    assert cases and all("withheld_future_correct_mask_16" in z for z in cases)
    best_order=None
    best_key=None
    searched=0
    for order in permutations(range(PUBLIC_TEMPLATE_COUNT),PAID_CHECK_BUDGET):
        scored=score_unique_signatures(cases,order)
        # Deterministic ties: correctness, false accepts, fewer checks,
        # lexicographically smaller order.
        key=(scored["future_correct"],-scored["false_accept"],
             -scored["paid_checks"],tuple(-x for x in order))
        searched+=1
        if best_key is None or key>best_key:
            best_key=key;best_order=order
    assert searched==TOTAL_ORDERS
    return {"ordered_top4":list(best_order),
            "oracle_selection_used_withheld_future_grades":True,
            "orders_exhaustively_checked":searched,
            **score_order(cases,best_order)}


def mask_conditional_hindsight(cases):
    groups=defaultdict(list)
    for row in cases:
        groups[row["public_accepted_mask_16"]].append(row)
    best=0
    details=[]
    for mask,items in sorted(groups.items()):
        possibilities=[j for j in range(16) if mask & (1<<j)]
        if not possibilities:
            chosen=None;value=0
        else:
            counts={j:sum(bool(z["withheld_future_correct_mask_16"]&(1<<j))
                          for z in items) for j in possibilities}
            chosen=max(counts,key=lambda j:(counts[j],-j))
            value=counts[chosen]
        best+=value
        details.append({"public_accepted_mask_16":mask,"cases":len(items),
                        "hindsight_best_candidate":chosen,
                        "hindsight_future_correct":value})
    per_case_clairvoyant=sum(bool(z["withheld_future_correct_mask_16"])
                            for z in cases)
    assert best<=per_case_clairvoyant<=len(cases)
    return {"public_mask_equivalence_groups":len(groups),
            "hindsight_correct_with_one_template_per_public_mask":best,
            "per_case_any_of_sixteen_future_correct_upper_bound":per_case_clairvoyant,
            "groups":details,
            "uses_full_sixteen_public_oracle_bits":True,
            "uses_withheld_future_grades_for_diagnostic_best_choice":True}


def run(dev_cases=DEV_CASES,test_cases=TEST_CASES):
    if dev_cases!=DEV_CASES or test_cases!=TEST_CASES:
        raise ValueError("NOR15C frozen dev/test cohort requirements")
    ck,probe,negative,previous_weights,original,design,roster=nor13.frozen_data()
    assert ck["parameters"]==1000 and old.BASIN_SHA==nor13.READOUT_SHA
    seen=all_prior_source_fingerprints(roster)
    dev_items=late_source_new(seen,dev_cases,DEV_OFFSET,"dev")
    dev=[observe_one(z,ck) for z in dev_items]
    assert len({x["public_features_sha256"] for x in dev})==1
    dev_order=optimize_order(dev)
    # TEST has not entered the dev order optimizer.
    test_items=late_source_new(seen,test_cases,TEST_OFFSET,"test")
    test=[observe_one(z,ck) for z in test_items]
    assert len({x["public_features_sha256"] for x in test})==1
    assert dev[0]["public_features_sha256"]==test[0]["public_features_sha256"]
    assert len({x["public_visible_sha256"] for x in dev+test})==dev_cases+test_cases
    original_orders={tuple(z["original_order_16"]) for z in dev+test}
    assert len(original_orders)==1
    original_order=list(next(iter(original_orders)))[:4]
    original_eval=score_order(test,original_order)
    chosen_eval=score_order(test,dev_order["ordered_top4"])
    random_order=list(range(16))
    random.Random(RANDOM_ORDER_SEED).shuffle(random_order)
    random_order=random_order[:4]
    random_eval=score_order(test,random_order)
    hindsight=optimize_order(test)
    full_mask=mask_conditional_hindsight(test)
    assert hindsight["future_correct"]>=max(
        original_eval["future_correct"],chosen_eval["future_correct"],
        random_eval["future_correct"])
    return {
        "schema":SCHEMA,
        "source_checkpoint_sha256":old.BASIN_SHA,
        "frozen_probe_sha256":design["frozen_probe_json_sha256"],
        "train_manifest":design["manifests"]["train"],
        "new_source_family":FAMILY,
        "dev_seed_offset":DEV_OFFSET,"test_seed_offset":TEST_OFFSET,
        "dev_cases":len(dev),"test_cases":len(test),
        "distinct_public_visible_sources":len({z["public_visible_sha256"] for z in dev+test}),
        "unique_model_visible_feature_hashes":len({
            z["public_features_sha256"] for z in dev+test}),
        "common_model_visible_feature_sha256":test[0]["public_features_sha256"],
        "original_shared_four_order":original_order,
        "development_future_selected_four_order":dev_order,
        "test_original_shared_model_policy":original_eval,
        "test_dev_selected_diagnostic_policy":chosen_eval,
        "test_seeded_random_shared_policy":{"order":random_order,**random_eval},
        "test_oracle_hindsight_optimal_four_order":hindsight,
        "test_full_mask_group_hindsight_bounds":full_mask,
        "separate_diagnostic_public_16_template_checks":16*(dev_cases+test_cases),
        "separate_diagnostic_withheld_future_16_grades":16*(dev_cases+test_cases),
        "at_most_paid_native_four_checks_per_test_case":True,
        "dev_source_manifest_sha256":sha256("\n".join(
            z["public_visible_sha256"] for z in dev).encode()).hexdigest(),
        "test_source_manifest_sha256":sha256("\n".join(
            z["public_visible_sha256"] for z in test).encode()).hexdigest(),
        "dev_case_records":dev,"test_case_records":test,
        "scope":{
            "nonadaptive_shared_feature_order_class_only":True,
            "all_oracle_future_labels_scoring_or_dev_diagnostic_only":True,
            "test_future_labels_absent_from_dev_selection":True,
            "hindsight_test_oracle_is_nondeployable":True,
            "late_source_input_fibre_aliases_even_across_new_visible_sources":True,
            "does_not_certify_population_ceiling":True,
            "does_not_establish_conditional_mutual_information":True,
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dev-cases",type=int,default=DEV_CASES)
    p.add_argument("--test-cases",type=int,default=TEST_CASES)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=run(args.dev_cases,args.test_cases)
    if args.check:
        assert result["source_checkpoint_sha256"]==old.BASIN_SHA
        assert result["dev_cases"]==80 and result["test_cases"]==160
        assert result["distinct_public_visible_sources"]==240
        assert result["unique_model_visible_feature_hashes"]==1
        assert result["development_future_selected_four_order"]["orders_exhaustively_checked"]==43680
        assert result["test_oracle_hindsight_optimal_four_order"]["orders_exhaustively_checked"]==43680
        assert result["separate_diagnostic_public_16_template_checks"]==3840
        assert result["separate_diagnostic_withheld_future_16_grades"]==3840
        assert result["test_original_shared_model_policy"]["paid_checks"]<=640
        assert result["test_dev_selected_diagnostic_policy"]["paid_checks"]<=640
    encoded=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(encoded)
    else:
        print(encoded,end="")


if __name__=="__main__":
    main()
