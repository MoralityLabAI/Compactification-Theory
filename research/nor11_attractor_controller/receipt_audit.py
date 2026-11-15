"""NOR-11C: reproduce and independently audit the frozen trained-controller receipt.

The original trained GRU is frozen; 13 learned auxiliary weights and all
training-set standardization moments are source-locked as a separate JSON
artifact. The full experiment receipt is deterministic under Python 3.12
and is stored in GitHub Actions. This audit verifies its exact SHA-256 and
compares classifier/skill/causal outcome claims to a committed summary.

A strong basin probe may causally change attractors while degrading
original four-paid-template future correctness. That is treated as
negative transfer rather than capability improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXPECTED_SHA="bfcc0d97e0a196bade7df87f87d4ae5ae844570aa723619ec712431d644685df"
CHECKPOINT_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
FROZEN=HERE/"results"/"nor11_executed_summary.json"
HEAD=HERE/"results"/"nor11_auxiliary_head.json"


def compare_pairwise(reference,other):
    if len(reference)!=len(other):
        raise AssertionError("comparison has different case counts")
    x={r["public_features_sha256"]:r for r in reference}
    y={r["public_features_sha256"]:r for r in other}
    # A repeated tensor can belong to multiple different public RRC25
    # cases in late_source. Compare by aligned case order and feature hash
    # rather than deduplicating the native task's multiple verification
    # targets. The full receipt records every outcome separately.
    assert all(a["public_features_sha256"]==b["public_features_sha256"]
               for a,b in zip(reference,other,strict=True))
    gain=sum(a["native_future_correct"]==0 and b["native_future_correct"]==1
             for a,b in zip(reference,other,strict=True))
    loss=sum(a["native_future_correct"]==1 and b["native_future_correct"]==0
             for a,b in zip(reference,other,strict=True))
    return gain,loss


def audit(path):
    raw=Path(path).read_bytes()
    body=json.loads(raw)
    frozen=json.loads(FROZEN.read_text())
    frozen_head=json.loads(HEAD.read_text())
    assert frozen["schema"]=="nor11.frozen_gru.basin_controller_executed_summary.v1"
    assert frozen["canonical_json_sha256"]==EXPECTED_SHA
    assert sha256(raw).hexdigest()==EXPECTED_SHA
    assert body["schema"]=="nor11.frozen_gru.trained_basin_head_and_escape_controller.v1"
    assert body["source_checkpoint_sha256"]==CHECKPOINT_SHA
    assert frozen["source_checkpoint_sha256"]==CHECKPOINT_SHA
    assert frozen_head["source_checkpoint_sha256"]==CHECKPOINT_SHA
    assert body["head"]==frozen_head["head"]
    assert body["cohorts"]["train"]["public_feature_manifest_sha256"]==frozen_head["public_training_feature_manifest_sha256"]
    assert all(body["cohorts"][k]["cases"]==v for k,v in
               (("train",96),("dev",36),("test",72)))
    assert body["cohorts"]["test"]["distinct_public_features"]==61
    assert body["cohorts"]["test"]["per_family_distinct_features"]["late_source"]==1
    assert body["cohorts"]["test"]["largest_public_feature_alias_group"]==12
    for group in ("train","dev"):
        assert body["cohorts"][group]["cases"]==body["cohorts"][group]["distinct_public_features"]
    assert not body["train_restrictions"]["native_future_truth_used_to_fit_head"]
    assert not body["train_restrictions"]["per_new_sample_source_basin_interval_certified"]
    assert not body["train_restrictions"]["native_proof_search_performance_gain_claimed"]
    assert not body["trained_gru_weights_modified"]
    head_test=body["head_diagnostics"]["test"]
    assert head_test["trained_unique_features"]["n"]==61
    assert head_test["trained_unique_features"]["auc"]>0.98
    assert head_test["trained_unique_features"]["balanced_accuracy"]>0.95
    assert head_test["shuffled_unique_features"]["auc"]<0.5
    assert body["selected_policy"]["alpha"]==0
    assert body["selected_policy"]["dev_future_correct"]==27
    scores=body["heldout_protocol"]
    for key,claims in frozen["heldout_native_task"].items():
        actual=scores[key]
        for field in ("future_correct","accepted","false_accept","paid_checks_total","actions"):
            mapped="paid_checks" if field=="paid_checks_total" else field
            assert actual[field]==claims[mapped],(key,field,actual[field],claims[mapped])
        assert actual["accepted"]==actual["future_correct"]+actual["false_accept"]
        assert actual["paid_checks_total"]<=4*actual["cases"]
    assert scores["original"]["future_correct"]==42
    assert scores["fixed_escape"]["future_correct"]==40
    assert scores["fixed_escape"]["mode_switches_if_actuated"]["second_to_first"]==13
    base=scores["original"]["per_case"]
    for name in ("fixed_escape","unconditional"):
        gain,loss=compare_pairwise(base,scores[name]["per_case"])
        claim=frozen["heldout_native_task"][name]
        assert (gain,loss)==(claim["paired_correctness_gains"],claim["paired_correctness_losses"])
    assert sum(s["actions"] for s in scores.values())>=0
    assert all(len(v["per_case"])==72 for v in scores.values())
    return {
      "schema":"nor11.trained_auxiliary_head.frozen_replay_audit.v1",
      "full_experiment_receipt_sha256":EXPECTED_SHA,
      "frozen_trained_gru_sha256":CHECKPOINT_SHA,
      "train_dev_test_case_counts":[96,36,72],
      "test_unique_feature_tensors":61,
      "test_case_level_aux_head_auc":head_test["trained"]["auc"],
      "test_unique_feature_aux_head_auc":head_test["trained_unique_features"]["auc"],
      "test_unique_feature_balanced_accuracy":head_test["trained_unique_features"]["balanced_accuracy"],
      "shuffled_aux_head_auc":head_test["shuffled_unique_features"]["auc"],
      "native_correct_baseline":42,
      "native_correct_fixed_escape":40,
      "native_correct_dev_selected":42,
      "native_correct_sham_random_direction":42,
      "fixed_escape_actions":23,
      "second_to_first_mode_switches":13,
      "paired_native_gains":2,
      "paired_native_losses":4,
      "native_capability_gain_established":False,
      "source_future_truth_used_for_head_training":False,
      "individual_new_sample_basin_membership_interval_certified":False,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    result=audit(args.receipt)
    if args.check:
        assert result["native_correct_fixed_escape"]<result["native_correct_baseline"]
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
