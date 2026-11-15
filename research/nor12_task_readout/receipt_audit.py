"""NOR-12C: exact audit of frozen training and independent-confirmation receipts.

This is an independent arithmetic/source-provenance audit of two GitHub
Actions results, not a substitute for re-running their full CPU training
and public exact-verifier test campaign. All study claims remain empirical.
"""
from __future__ import annotations

from fractions import Fraction as Q
from hashlib import sha256
import json
from math import comb,isclose
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIRST=HERE/"results"/"nor12_initial_receipt.json"
SECOND=HERE/"results"/"nor12b_confirmation_summary.json"
CK=ROOT/"research"/"rrc_compactification"/"results"/"rrc25_recurrent.json"
PROBE=ROOT/"research"/"nor11_attractor_controller"/"results"/"nor11_auxiliary_head.json"


def paired_exact_p(gains,losses):
    n=gains+losses
    if n==0:
        return Q(1)
    return min(Q(1), Q(2*sum(comb(n,k) for k in range(min(gains,losses)+1)),2**n))


def audit():
    first=json.loads(FIRST.read_text())
    second=json.loads(SECOND.read_text())
    checkpoint=sha256(CK.read_bytes()).hexdigest()
    probe=sha256(PROBE.read_bytes()).hexdigest()
    assert first["schema"]=="nor12.public_demo_supervised_residual_readout.frozen_first_run.v1"
    assert second["schema"]=="nor12b.independent_holdout_confirmation.frozen_summary.v1"
    assert checkpoint==first["source_checkpoint_sha256"]==second["source_checkpoint_sha256"]
    assert probe==first["frozen_probe_json_sha256"]
    assert first["full_result_json_sha256"]==second["initial_full_result_sha256"]
    assert second["confirmation_result_sha256"]=="70e624e5b1b40d798c398bb1a35e804569557bcb49c88ef18e29454135d0e6e0"
    assert first["independent_confirmation_completed"] is False
    assert first["training"]["public_oracle_train_checks"]==1536
    assert second["training_and_dev_frozen"]["training_full_public_oracle_checks"]==1536
    assert second["training_and_dev_frozen"]["head_weight_digests_verified_before_confirmation"]
    assert second["training_and_dev_frozen"]["scales"]==first["fixed_development_scales"]
    assert set(first["fitted_weight_sha256_by_kind"])=={
      "basin","shuffled_basin","random_feature","hidden_only"}
    assert all(len(x)==64 for x in first["fitted_weight_sha256_by_kind"].values())
    assert second["confirmation"]["total_cases"]==4*72==288
    assert second["confirmation"]["unique_model_input_tensors_overall"]==241
    assert second["confirmation"]["unique_tensors_per_rep"]==61
    raw=second["confirmation"]["total_original"]
    assert (raw["future_correct"],raw["false_accept"],raw["paid_checks"])==(160,26,809)
    controls=second["confirmation"]["alternatives"]
    verified={}
    for name,record in controls.items():
        assert record["future_correct"]==raw["future_correct"]+record["gains"]-record["losses"]
        assert record["false_accept"]>=0 and record["paid_checks"]<=288*4
        p=paired_exact_p(record["gains"],record["losses"])
        assert isclose(float(p),record["case_level_exact_mcnemar_p"],abs_tol=1e-15)
        verified[name]={
            "future_correct":record["future_correct"],
            "net_case_gain":record["gains"]-record["losses"],
            "unadjusted_case_p":float(p)
        }
    assert controls["basin"]["future_correct"]==controls["random_feature"]["future_correct"]==167
    assert controls["basin"]["case_level_exact_mcnemar_p"]>0.05
    assert controls["basin"]["feature_group_sign_p"]>0.05
    assert controls["hidden_only"]["future_correct"]==controls["shuffled_basin"]["future_correct"]==160
    first_result=first["heldout_case_counts"]
    assert first_result["basin"]["future_correct"]==44
    assert first_result["original"]["future_correct"]==42
    assert first_result["basin"]["paired_gains"]==2
    assert first_result["basin"]["paired_losses"]==0
    ablation=second["first_test_posthoc_basin_feature_ablation"]
    assert {x["future_correct"] for x in ablation.values() if isinstance(x,dict)}=={44}
    assert ablation["original_test_gains_from_one_late_source_feature_group"]==2
    assert ablation["original_test_other_family_future_correct_changes"]==0
    assert second["statements"]["basin_specific_advantage_over_random_feature_established"] is False
    assert second["statements"]["causal_necessity_of_basin_scalar_in_first_test_established"] is False
    assert second["statements"]["training_uses_public_degree_192_exact_candidate_checks_only"] is True
    return {
      "schema":"nor12c.two_experiments.pinned_arithmetic_audit.v1",
      "checkpoint_sha256":checkpoint,
      "probe_sha256":probe,
      "first_run_sha256":first["full_result_json_sha256"],
      "independent_confirmation_sha256":second["confirmation_result_sha256"],
      "fresh_cases":288,
      "fresh_unique_feature_tensors":241,
      "original_future_correct":160,
      "control_scores":verified,
      "first_split_basin_q_ablation_changes_correct_count":False,
      "specific_basin_probability_value_proven_beneficial":False,
      "native_task_generalization_confirmed_beyond_null":False,
    }


def main():
    print(json.dumps(audit(),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
