"""Independent NOR-15B native four-check receipt and 8320-decision audit.

No import from any experimental runner. Full evidence mode checks the exact
archived source bytes, case-level native verification records, 1000 distinct
nonlate input hashes, per-family arithmetic and all six paired sign tests.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor15b_frozen_receipt.json"
PINNED_SHA="1f5f068b522780e66d639d0429369c66775cedbef997f510824be124d4cde124"
MODEL_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def exact_sign(a,b):
    n=a+b
    if n==0:return 1.
    return min(1.,2*sum(comb(n,k) for k in range(min(a,b)+1))/2**n)


def paired(left,right):
    groups=defaultdict(int)
    gain=loss=cases=0
    for a,b in zip(left,right,strict=True):
        assert a["public_features_sha256"]==b["public_features_sha256"]
        assert a["family"]==b["family"]
        if a["family"]=="late_source":continue
        cases+=1
        d=b["future_correct"]-a["future_correct"]
        assert d in (-1,0,1)
        gain+=d>0;loss+=d<0
        groups[a["public_features_sha256"]]+=d
    pos=sum(x>0 for x in groups.values())
    neg=sum(x<0 for x in groups.values())
    return {"cases":cases,"unique_public_inputs":len(groups),
            "paired_gains":gain,"paired_losses":loss,
            "net":gain-loss,"positive_input_groups":pos,
            "negative_input_groups":neg,
            "unadjusted_exact_sign_p":exact_sign(pos,neg)}


def frozen():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor15b.frozen_independent_1040_case_native_task_receipt.v1"
    assert r["full_json_sha256"]==PINNED_SHA
    assert r["full_json_uncompressed_bytes"]==3224589
    assert r["source_checkpoint_sha256"]==MODEL_SHA
    assert r["train_cases"]==96 and r["dev_cases"]==36
    assert r["train_public_template_checks"]==1536
    assert r["task_parameters_allocated_per_head"]==400
    assert r["test_cases"]==1040 and r["primary_nonlate_distinct_inputs"]==1000
    assert r["secondary_late_source_distinct_inputs"]==1
    assert r["global_dev_selection"]["kind"]=="shuffled_early"
    assert r["outcomes"]["original"]==[639,20,2943,631]
    assert r["outcomes"]["residual_early"]==[666,21,2906,660]
    assert r["outcomes"]["endpoint_nonlinear"]==[669,21,2938,662]
    assert not r["primary_specific_trajectory_hypothesis_passed"]
    return r


def inspect_full(data,r):
    assert data["schema"]=="nor15b.public_verified_trajectory_innovation.v1"
    assert data["source_checkpoint_sha256"]==MODEL_SHA
    assert data["frozen_probe_sha256"]==r["frozen_probe_sha256"]
    assert data["original_train_manifest"]==r["train_manifest"]
    assert data["original_dev_manifest"]==r["dev_manifest"]
    assert data["source_cases_train"]==96 and data["source_cases_dev"]==36
    assert data["shared_public_train_template_checks"]==1536
    assert data["per_model_public_label_exposures"]==1536
    assert data["trained_readout_slots"]==400
    assert data["training_epochs"]==150
    assert data["repetitions"]==4 and data["nonlate_per_family_per_repetition"]==50
    assert data["late_source_per_repetition"]==10
    assert data["dev_global_selected"]==r["global_dev_selection"]
    assert data["bonferroni_three_control_threshold"]==r["primary_success_threshold"]
    assert not data["residual_early_passed_preregistered_primary"]
    assert data["scope"]["no_private_future_used_for_train"]
    assert data["scope"]["source_input_and_feature_disjoint_previous_NOR11_12_13_14_15A"]
    assert data["scope"]["single_late_source_hash_secondary"]
    assert data["scope"]["test_inference_four_paid_checks"]
    assert len(data["cohorts"])==4
    assert [z["public_features_manifest_sha256"] for z in data["cohorts"]
            ]==r["test_public_features_manifests"]
    assert [z["source_visible_manifest_sha256"] for z in data["cohorts"]
            ]==r["test_source_visible_manifests"]
    assert data["model_weights_sha256"]==r["model_weight_sha256"]
    assert data["reconstructor"]["dev_selection"]["nonlinear"]["lambda"]==.001
    expected_records=set(r["outcomes"])
    assert set(data["records"])==expected_records
    assert set(data["aggregated"])==expected_records
    reference=data["records"]["original"]
    assert len(reference)==1040
    nonlate=[z for z in reference if z["family"]!="late_source"]
    late=[z for z in reference if z["family"]=="late_source"]
    assert len(nonlate)==1000 and len(late)==40
    assert len({z["public_features_sha256"] for z in nonlate})==1000
    assert len({z["public_features_sha256"] for z in late})==1
    assert len({z["public_features_sha256"] for z in reference})==1001
    families=Counter(z["family"] for z in reference)
    assert set(families)=={"signed","sparse","product","rational","squared","late_source"}
    assert families["late_source"]==40
    assert set(v for k,v in families.items() if k!="late_source")=={200}

    fields=r["outcome_fields"]
    assert fields==["all_future_correct","all_false_accept","all_paid_checks","nonlate_correct"]
    for name,records in data["records"].items():
        assert len(records)==1040
        for a,b in zip(reference,records,strict=True):
            assert a["public_features_sha256"]==b["public_features_sha256"]
            assert a["family"]==b["family"]
            assert b["accepted"] in (0,1) and b["future_correct"] in (0,1)
            assert b["false_accept"] in (0,1)
            assert b["accepted"]==b["future_correct"]+b["false_accept"]
            assert 1<=b["paid_checks"]<=4
            assert len(b["verification_trace"])==b["paid_checks"]
            assert b["top1"]==b["verification_trace"][0]
            assert b["verified_class"]==(b["verification_trace"][-1] if b["accepted"] else None)
        totals=[sum(z["future_correct"] for z in records),
                sum(z["false_accept"] for z in records),
                sum(z["paid_checks"] for z in records),
                sum(z["future_correct"] for z in records if z["family"]!="late_source")]
        assert totals==r["outcomes"][name],(name,totals)
        aggregate=data["aggregated"][name]
        assert aggregate["cases"]==1040
        assert aggregate["nonlate_cases"]==1000
        assert aggregate["nonlate_unique_inputs"]==1000
        assert aggregate["future_correct"]==totals[0]
        assert aggregate["false_accept"]==totals[1]
        assert aggregate["paid_checks"]==totals[2]
        assert aggregate["nonlate_correct"]==totals[3]
        assert aggregate["nonlate_false_accept"]==sum(
            z["false_accept"] for z in records if z["family"]!="late_source")
        for family,details in aggregate["by_family"].items():
            group=[z for z in records if z["family"]==family]
            assert details["cases"]==len(group)
            assert details["future_correct"]==sum(z["future_correct"] for z in group)
            assert details["false_accept"]==sum(z["false_accept"] for z in group)
        for i,cohort in enumerate(data["cohorts"]):
            chunk=records[i*260:(i+1)*260]
            assert len(chunk)==260
            digest=sha256("\n".join(z["public_features_sha256"]
                                    for z in chunk).encode()).hexdigest()
            assert digest==cohort["public_features_manifest_sha256"]
            recorded=cohort["arms"][name]
            assert recorded["future_correct"]==sum(z["future_correct"] for z in chunk)
            assert recorded["false_accept"]==sum(z["false_accept"] for z in chunk)
            assert recorded["paid_checks"]==sum(z["paid_checks"] for z in chunk)

    test_pairs={
        "residual_vs_original":("original","residual_early"),
        "residual_vs_reconstructed":("reconstructed_early","residual_early"),
        "residual_vs_endpoint_nonlinear":("endpoint_nonlinear","residual_early"),
        "residual_vs_shuffled_early":("shuffled_early","residual_early"),
        "residual_vs_hidden_only":("hidden_only","residual_early"),
        "raw_vs_residual":("residual_early","raw_early")}
    assert set(data["predeclared_primary_paired_signs"])==set(test_pairs)
    for name,(left,right) in test_pairs.items():
        calc=paired(data["records"][left],data["records"][right])
        expected=data["predeclared_primary_paired_signs"][name]
        receipt=r["primary_paired"][name]
        assert calc["cases"]==calc["unique_public_inputs"]==1000
        for key in calc:
            if key=="unadjusted_exact_sign_p":
                assert abs(calc[key]-expected[key])<1e-13
                assert abs(calc[key]-receipt["p"])<1e-13
            else:
                assert calc[key]==expected[key],(name,key)
        assert calc["paired_gains"]==receipt["gains"]
        assert calc["paired_losses"]==receipt["losses"]
    return {
        "cases":1040,
        "primary_distinct_nonlate":1000,
        "native_decision_records_checked":sum(len(v) for v in data["records"].values()),
        "paired_exact_signs_checked":len(test_pairs),
        "cohorts_checked":4,
        "preregistered_trajectory_specific_hypothesis_passed":False}


def main():
    parser=argparse.ArgumentParser()
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument("--full",type=Path)
    inputs.add_argument("--archive",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    r=frozen()
    raw=None
    if args.full:raw=args.full.read_bytes()
    if args.archive:
        with ZipFile(args.archive) as z:
            assert z.namelist()==["nor15b_full.json"],z.namelist()
            assert z.testzip() is None
            raw=z.read("nor15b_full.json")
    if raw is not None:
        assert len(raw)==r["full_json_uncompressed_bytes"]
        assert sha256(raw).hexdigest()==PINNED_SHA
        result=inspect_full(json.loads(raw),r)
        print("NOR15B INDEPENDENT NATIVE RECEIPT AUDIT PASSED",result)
    else:
        print("NOR15B FIRST FULL 1040-CASE RECEIPT HASH VALID")
    if args.check:
        assert r["first_hosted_run"].startswith("https://github.com/")


if __name__=="__main__":
    main()
