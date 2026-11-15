"""NOR-13R: frozen-source arithmetic and epistemic-scope receipt audit.

Validates the published exact run receipt against the original RRC-25
checkpoint bytes, closed-form McNemar arithmetic, matched four-check
budgets, alias-aware input totals and fixed-weight gate substitutions.

An optional path supplies the full prior GitHub Actions JSON for byte-
identical replay. Without it this checks the committed source-bound
summary only; it does NOT recompute task training or held-out outcomes.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from hashlib import sha256
from math import comb,isclose
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RECEIPT=HERE/"results"/"nor13r_frozen_summary.json"
CHECKPOINT=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def exact_two_sided_binomial(gains,losses):
    if gains<0 or losses<0:
        raise ValueError("invalid discordant count")
    n=gains+losses
    if n==0:
        return Q(1)
    return min(Q(1),Q(2*sum(comb(n,k) for k in range(min(gains,losses)+1)),2**n))


def audit(full_json_path=None):
    frozen=json.loads(RECEIPT.read_text())
    assert frozen["schema"]=="nor13r.independent_gate_interaction_and_rank_receipt.v1"
    assert sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA==frozen["source_checkpoint_sha256"]
    assert frozen["trained_real_GRU_parameters"]==1000
    assert frozen["trained_auxiliary_basin_head_parameters"]==13
    assert frozen["residual_parameter_slots_per_head"]==16*25==400
    assert frozen["public_training_source_cases"]==96
    assert frozen["public_development_cases"]==36
    assert frozen["public_template_checks_per_trained_head"]==16*96
    assert frozen["fresh_holdout_cases"]==288
    assert frozen["model_visible_unique_feature_clusters"]==241
    assert 4*frozen["unique_feature_clusters_per_repetition"]==244
    assert frozen["development_global_frozen_choice"]["kind"]=="random_gate"
    assert frozen["development_local_scales"]["basin_gate"]==.5
    scores=frozen["holdout_results"]
    baseline=scores["original"]
    assert (baseline["future_correct"],baseline["false_accept"],baseline["paid_checks"])==(168,25,809)
    assert all(v["future_correct"]<=288 and v["paid_checks"]<=1152 for v in scores.values())
    assert scores["basin_gate"]["future_correct"]==scores["random_gate"]["future_correct"]==171
    for key in ("basin_gate","random_gate","shuffled_gate","constant_gate"):
        x=scores[key]
        assert x["future_correct"]-baseline["future_correct"]==x["paired_gains"]-x["paired_losses"]
    case_p=exact_two_sided_binomial(scores["basin_gate"]["paired_gains"],scores["basin_gate"]["paired_losses"])
    assert isclose(float(case_p),scores["basin_gate"]["unadjusted_case_mcnemar_p"],abs_tol=1e-15)
    assert case_p>Q(1,20)
    ab=frozen["fixed_weight_real_basin_gate_ablation_over_all_288_cases"]
    ref=ab["trained_gate"]
    assert ref["future_correct"]==scores["basin_gate"]["future_correct"]
    assert ab["gate_replaced_by_random_hash"]["future_correct"]>ref["future_correct"]
    for key,x in ab.items():
        assert x["future_correct"]-ref["future_correct"]==(
          x["paired_gains_vs_trained_gate"]-x["paired_losses_vs_trained_gate"])
    geometry=frozen["exact_binary_rational_gate_endpoint_order_audit"]
    assert geometry["total_group_replicates"]==244
    assert sum(geometry["ordered_top4_invariant_input_counts_per_replicate"])==28
    assert all(v==0 for v in geometry["fully_rank_invariant_input_counts_per_replicate"])
    assert geometry["floating_point_rounding_certified"] is False
    assert frozen["scope"]["marginal_basin_gate_causally_changes_some_verified_cases"]
    assert frozen["scope"]["basin_specific_superiority_to_matched_sham"] is False
    assert frozen["scope"]["private_future_coefficients_used_for_training"] is False
    assert frozen["scope"]["source_model_weights_modified"] is False

    if full_json_path:
        raw=Path(full_json_path).read_bytes()
        digest=sha256(raw).hexdigest()
        assert digest==frozen["full_github_actions_json_sha256"]
        assert len(raw)==frozen["full_json_uncompressed_bytes"]
        full=json.loads(raw)
        assert full["source_checkpoint_sha256"]==SHA
        assert full["matched_parameter_slots_per_head"]==400
        assert full["scope"]["future_labels_used_for_training"] is False
        assert full["dev_model_selection"]["kind"]=="random_gate"
        for key,x in scores.items():
            y=full["aggregated"][("dev_global" if key=="global_dev_choice" else key)]
            for field in ("future_correct","false_accept","paid_checks"):
                assert x[field]==y[field]
        for key,x in ab.items():
            y=full["all_fresh_cohort_fixed_weight_gate_ablations"][key]
            assert x["future_correct"]==y["future_correct"]
            assert x["false_accept"]==y["false_accept"]
        assert sum(r["top4_gate_invariant_unique_inputs"] for r in
            full["all_fresh_cohort_rank_stability"])==28

    return {
      "schema":"nor13r.source_locked_frozen_receipt_audit.v1",
      "source_checkpoint_sha256":SHA,
      "full_result_sha256":frozen["full_github_actions_json_sha256"],
      "all_288_case_observed_task_gain":scores["basin_gate"]["future_correct"]-baseline["future_correct"],
      "matched_random_gate_score":scores["random_gate"]["future_correct"],
      "case_level_mcnemar_two_sided_p":float(case_p),
      "true_gate_versus_zero_net_correct":ref["future_correct"]-ab["gate_zero"]["future_correct"],
      "true_gate_versus_random_replacement_net_correct":ref["future_correct"]-ab["gate_replaced_by_random_hash"]["future_correct"],
      "rational_lift_ordered_top4_invariant_group_replicates":28,
      "rational_lift_total_group_replicates":244,
      "verified_full_artifact":bool(full_json_path),
      "basin_specific_skill_gain_established":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full-json",type=Path)
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    output=audit(a.full_json)
    if a.check:
        assert not output["basin_specific_skill_gain_established"]
    print(json.dumps(output,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
