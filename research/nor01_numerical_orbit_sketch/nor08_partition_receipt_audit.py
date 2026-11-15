"""NOR-08C: audit frozen basin mode and sign-flip evaluation receipt.

Exact integer contingency arithmetic only. The enrichment statistic is
post hoc exploratory: the trained-GRU experiment was selected because
one squared-family example previously reached the second attractor.
"""
from __future__ import annotations

import json
from fractions import Fraction as Q
from math import comb
from pathlib import Path

HERE=Path(__file__).resolve().parent
FROZEN=HERE/"results"/"nor08_heldout_partition_summary.json"
CHECKPOINT_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def exact_p_tail(N,K,n,x):
    if not 0<=K<=N or not 0<=n<=N or not 0<=x<=n:
        raise ValueError("invalid hypergeometric counts")
    return sum((Q(comb(K,k)*comb(N-K,n-k),comb(N,n))
                for k in range(x,min(n,K)+1)
                if 0<=n-k<=N-K),Q(0))


def audit():
    r=json.loads(FROZEN.read_text())
    assert r["schema"]=="nor08c.heldout_partition.frozen_summary.v1"
    assert r["trained_checkpoint_sha256"]==CHECKPOINT_SHA
    assert r["samples_per_family"]==12 and r["cohort_size"]==72
    matrix=r["contingency"]
    assert set(matrix)=={"signed","sparse","product","rational","squared","late_source"}
    assert all(sum(row.values())==12 for row in matrix.values())
    total_first=sum(x["first"] for x in matrix.values())
    total_second=sum(x["second"] for x in matrix.values())
    unresolved=sum(x["unresolved"] for x in matrix.values())
    assert (total_first,total_second,unresolved)==(45,27,0)
    assert r["mode_totals"]=={"first":45,"second":27,"unresolved":0}
    accurate=matrix["squared"]["second"]+sum(row["first"]
                                               for key,row in matrix.items() if key!="squared")
    assert Q(accurate,72)==Q(r["squared_versus_rest_fixed_mode_rule_accuracy"])
    assert Q(r["majority_family_baseline_accuracy"])==Q(5,6)
    transitions=r["intervention_mode_changes"]
    assert sum(transitions.values())==72
    assert transitions["same_mode"]==70 and transitions["other"]==0
    # Squared-family second labels: 9/12; all other families: 18/60.
    odds=Q(matrix["squared"]["second"]*
           sum(row["first"] for key,row in matrix.items() if key!="squared"),
           matrix["squared"]["first"]*
           sum(row["second"] for key,row in matrix.items() if key!="squared"))
    assert odds==Q(r["posthoc_squared_second_enrichment_odds_ratio"])==7
    p=exact_p_tail(72,27,12,9)
    assert p==Q(r["posthoc_one_sided_hypergeometric_p_fraction"])
    # These raw case-level counts overstate effective sample size:
    # 12 late_source cases share one exact 9x8 source feature tensor and
    # one exact 12D initial hidden state. Count their unique input once.
    diversity=r["effective_input_diversity"]
    fam=diversity["source_family_feature_counts"]
    assert diversity["unique_public_feature_tensors"]==61
    assert diversity["unique_native_initial_hidden_states"]==61
    assert fam["late_source"]=={"visible":12,"features":1,"hidden":1}
    assert all(fam[name]["features"]==12 and fam[name]["hidden"]==12
               for name in fam if name!="late_source")
    assert diversity["deduplicated_feature_unit_count"]==61
    dedup_first=total_first-(matrix["late_source"]["first"]-1)
    dedup_second=total_second
    assert diversity["deduplicated_mode_totals"]=={"first":34,"second":27}
    assert (dedup_first,dedup_second)==(34,27)
    dedup_correct=accurate-11
    assert Q(dedup_correct,61)==Q(diversity["deduplicated_squared_vs_rest_rule_accuracy"])
    assert Q(diversity["deduplicated_majority_family_baseline_accuracy"])==Q(49,61)
    dedup_odds=Q(9*(42-11),3*18)
    assert dedup_odds==Q(diversity["deduplicated_squared_second_enrichment_odds_ratio"])==Q(31,6)
    dedup_p=exact_p_tail(61,27,12,9)
    assert dedup_p==Q(diversity["deduplicated_posthoc_one_sided_hypergeometric_p_fraction"])
    assert r["retraining_performed"] is False
    assert r["source_future_labels_used"] is False
    assert r["source_to_second_basin_rational_capture_certified_by_this_probe"] is False
    return {"schema":"nor08c.heldout_partition.audit.v1",
            "cohort_size":72,
            "first_mode":45,"second_mode":27,
            "fixed_rule_accuracy":float(Q(accurate,72)),
            "majority_baseline":float(Q(5,6)),
            "intervention_label_switches":2,
            "posthoc_suggestive_squared_enrichment_odds_ratio":float(odds),
            "posthoc_unadjusted_fisher_right_tail":float(p),
            "effective_unique_feature_inputs":61,
            "effective_fixed_rule_accuracy":float(Q(40,61)),
            "effective_majority_baseline":float(Q(49,61)),
            "feature_deduped_exploratory_one_sided_tail":float(dedup_p),
            "scope":"72 visible-distinct cases have 61 unique feature tensors; exploratory post-hoc association, not a formal source-identity claim"}


def main():
    print(json.dumps(audit(),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
