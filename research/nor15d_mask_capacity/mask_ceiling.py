"""NOR-15D: exact hindsight ceiling when all 16 public accept bits are visible.

Exhaustively inspect all classes of identical accepted-template masks from
the FROZEN, ALREADY-EVALUATED NOR-15C source JSON. For each mask group A,
a policy observing only the full 16-bit mask must choose the same template
for every case in that group. The best hindsight choice is the candidate
maximising future-correct cases in that group. The group capacities add.

This is a retrospective, test-future-optimised oracle upper bound, not an
implementable trained policy or a limitation on richer public inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"nor15c_verifier_alias"/"results"/"nor15c_full.json"
SOURCE_SHA="f2dc2865fca360e4ccb717c0e84cde6fe5c1f4a1bea29111b5b1d250753e5f29"
SCHEMA="nor15d.full_public_acceptance_signature_hindsight_capacity.v1"


def group_capacity(rows):
    if not rows:raise ValueError("nonempty public mask group required")
    a=rows[0]["public_accepted_bitmask"]
    if a<=0 or a>=65536 or any(x["public_accepted_bitmask"]!=a for x in rows):
        raise ValueError("group must have identical nonempty public acceptance mask")
    allowed=[i for i in range(16) if (a>>i)&1]
    scores={str(j):sum((x["future_correct_bitmask_hindsight"]>>j)&1
                       for x in rows) for j in allowed}
    top=max(scores.values())
    best=min(int(j) for j,v in scores.items() if v==top)
    assert 0<=top<=len(rows)
    return {
        "public_accepted_bitmask":a,
        "public_accepted_template_count":len(allowed),
        "cases":len(rows),
        "oracle_selected_template":best,
        "oracle_future_correct":top,
        "unavoidable_future_incorrect_under_same_mask_choice":len(rows)-top,
        "candidate_future_correct_counts":scores,
        "source_visible_sha256":[x["public_visible_sha256"] for x in rows],
    }


def run(source=SOURCE):
    raw=Path(source).read_bytes()
    assert len(raw)==23205
    assert sha256(raw).hexdigest()==SOURCE_SHA
    d=json.loads(raw)
    assert d["schema"]=="nor15c.retrospective_four_check_fixed_order_oracle.v1"
    assert d["source_case_count"]==40 and d["unique_model_visible_feature_groups"]==1
    assert d["retrospective_hindsight_fixed_order_oracle"][
        "distinct_fixed_orders_exhaustively_checked"]==43680
    cases=d["case_receipts"]
    assert len(cases)==40
    assert len({x["public_visible_sha256"] for x in cases})==40
    assert len({x["public_features_sha256"] for x in cases})==1
    groups=defaultdict(list)
    for x in cases:
        assert (x["future_correct_bitmask_hindsight"] &
                ~x["public_accepted_bitmask"])==0
        groups[x["public_accepted_bitmask"]].append(x)
    capacities=[group_capacity(rows) for a,rows in sorted(groups.items(),
                key=lambda pair:(-len(pair[1]),pair[0]))]
    total=sum(x["oracle_future_correct"] for x in capacities)
    min_lost=sum(x["unavoidable_future_incorrect_under_same_mask_choice"]
                 for x in capacities)
    assert total+min_lost==len(cases)
    assert len(capacities)==d["unique_exact_public_acceptance_masks"]==14
    assert total==25 and min_lost==15
    biggest=capacities[0]
    assert biggest["cases"]==13 and biggest["public_accepted_bitmask"]==65535
    assert biggest["oracle_future_correct"]==3
    assert biggest["unavoidable_future_incorrect_under_same_mask_choice"]==10
    assert all(x["future_correct_bitmask_hindsight"]>0 for x in cases)
    base=d["original_fixed_four_outcome"]
    fixed=d["retrospective_hindsight_fixed_order_oracle"]["hindsight_optimal_metrics"]
    return {
        "schema":SCHEMA,
        "input_nor15c_json_sha256":SOURCE_SHA,
        "input_nor15c_original_source_cases":40,
        "input_one_model_visible_feature_group":True,
        "distinct_complete_public_masks":len(capacities),
        "original_fixed_four_future_correct":base["future_correct"],
        "best_hindsight_fixed_four_future_correct":fixed["future_correct"],
        "best_hindsight_full_mask_future_correct":total,
        "unavoidable_lost_future_correct_under_complete_mask_only":min_lost,
        "per_case_full_future_oracle_ceiling":sum(
            x["future_correct_bitmask_hindsight"]>0 for x in cases),
        "all_candidates_publicly_accepted_group":biggest,
        "total_public_mask_cases":sum(x["cases"] for x in capacities),
        "group_capacities":capacities,
        "capacity_integer_certificate":[x["oracle_future_correct"]
                                        for x in capacities],
        "scope":{
            "retrospective_existing_test_source_only":True,
            "full_sixteen_public_bits_seen_by_oracle":True,
            "future_labels_used_for_hindsight_group_choice":True,
            "not_deployable_model_evaluation":True,
            "mask_only_policy_same_choice_per_identical_mask":True,
            "randomized_mask_only_expected_success_at_most_deterministic_bound":True,
            "richer_public_source_observations_not_bounded":True,
            "not_conditional_mutual_information_certificate":True,
        }
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    x=run()
    if args.check:
        assert x["best_hindsight_full_mask_future_correct"]==25
        assert x["distinct_complete_public_masks"]==14
        assert x["all_candidates_publicly_accepted_group"]["cases"]==13
    raw=json.dumps(x,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(raw)
    else:print(raw,end="")


if __name__=="__main__":
    main()
