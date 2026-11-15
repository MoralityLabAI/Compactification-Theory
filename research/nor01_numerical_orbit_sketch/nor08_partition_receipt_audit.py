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
            "scope":"frozen empirical contingency, exploratory post-hoc association; source basin membership separate"}


def main():
    print(json.dumps(audit(),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
