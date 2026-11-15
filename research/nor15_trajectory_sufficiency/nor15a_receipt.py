"""Independent arithmetic audit of the frozen NOR-15A trajectory receipt.

Does not import the training, data-generation, or GRU code. Full mode reads
the original publicly archived JSON, checks its byte hash and every
individual reconstructed-state error and per-family total.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor15a_frozen_receipt.json"
SHA="9dbe31960a1866b8f4062cb148ab5db9c4b7743ea1fa28689b8377a78459f92b"
SOURCE="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def approx(x,y,tol=1e-11):
    return math.isclose(x,y,rel_tol=tol,abs_tol=tol)


def sign_p(a,b):
    n=a+b
    return 1. if n==0 else min(1.,2*sum(math.comb(n,i)
               for i in range(min(a,b)+1))/2**n)


def frozen_receipt():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor15a.frozen_trajectory_reconstruction_first_hosted_receipt.v1"
    assert r["first_full_output_sha256"]==SHA
    assert r["first_full_output_uncompressed_bytes"]==492261
    assert r["source_checkpoint_sha256"]==SOURCE
    assert r["train_cases"]==96 and r["dev_cases"]==36
    assert r["fresh_cases"]==288 and r["primary_nonlate_unique_inputs"]==240
    assert r["secondary_late_source_cases"]==48
    assert not r["conditional_information_established"]
    assert not r["task_correctness_observed"]
    assert r["primary_nonlate_standardized_MSE"]["nonlinear"]==.2845376993488323
    assert r["primary_paired_signs"]["nonlinear_vs_linear"]["first_lower_error"]==177
    return r


def validate_full(data,receipt):
    assert data["schema"]=="nor15a.frozen_gru.trajectory_endpoint_reconstructibility.v1"
    assert data["checkpoint_sha256"]==SOURCE
    assert data["frozen_probe_sha256"]==receipt["frozen_probe_sha256"]
    assert data["train_manifest"]==receipt["train_manifest"]
    assert data["dev_manifest"]==receipt["dev_manifest"]
    assert data["test_manifest_sha256"]==receipt["test_manifest"]
    assert data["train_cases"]==96 and data["dev_cases"]==36
    assert data["test_cases"]==288
    assert data["fresh_repetitions"]==4
    assert data["cases_per_family_per_repetition"]==12
    assert data["nonlate_unique_inputs"]==240 and data["late_source_unique_inputs"]==1
    assert data["scope"]["public_feature_only_recurrence"]
    assert data["scope"]["training_uses_hidden_labels_only_not_proof_futures"]
    assert data["scope"]["test_input_disjoint_all_NOR11_12_13_14_nonlate"]
    assert data["scope"]["finite_sample_reconstructibility_not_conditional_mutual_information"]
    assert data["scope"]["jacobian_is_finite_difference_F64_not_interval_certified"]
    assert data["development_frozen_selection"]["linear"]["lambda"]==.01
    assert data["development_frozen_selection"]["nonlinear"]["lambda"]==.001
    assert data["development_frozen_selection"]["shuffled_linear"]["lambda"]==10.
    assert set(data["outcomes"])==set(
        ("copy_endpoint","linear","mean","nonlinear","shuffled_linear"))
    assert len(data["records"])==288
    allrecords=data["records"]
    assert len(set(row["visible_sha256"] for row in allrecords))==288
    assert sha256("\n".join(z["digest"] for z in allrecords).encode()).hexdigest()==receipt["test_manifest"]
    nonlate=[r for r in allrecords if r["family"]!="late_source"]
    late=[r for r in allrecords if r["family"]=="late_source"]
    assert len(nonlate)==240 and len(late)==48
    assert len({r["digest"] for r in nonlate})==240
    assert len({r["digest"] for r in late})==1
    assert len({r["digest"] for r in allrecords})==241
    assert set(Counter(r["family"] for r in allrecords).values())=={48}
    xmap={r["digest"]:r for r in nonlate}
    assert len(xmap)==240
    for row in allrecords:
        assert len(row["h4"])==len(row["h9"])==12
        assert all(math.isfinite(z) for k in ("h4","h9") for z in row[k])
        assert set(row["standardized_reconstruction_MSE"])==set(data["outcomes"])
        assert all(math.isfinite(z) and z>=0
                   for z in row["standardized_reconstruction_MSE"].values())
    for name,outcome in data["outcomes"].items():
        m=statistics.fmean(z["standardized_reconstruction_MSE"][name]
                           for z in nonlate)
        med=statistics.median(z["standardized_reconstruction_MSE"][name]
                              for z in nonlate)
        total=statistics.fmean(z["standardized_reconstruction_MSE"][name]
                               for z in allrecords)
        assert approx(m,outcome["primary_mean_MSE"])
        assert approx(med,outcome["primary_median_MSE"])
        assert approx(total,outcome["all_case_mean_MSE"])
        assert approx(m,receipt["primary_nonlate_standardized_MSE"][name])
        assert outcome["primary_nonlate_cases"]==240
        for fam,value in outcome["by_family_mean_MSE"].items():
            observations=[z["standardized_reconstruction_MSE"][name]
                          for z in allrecords if z["family"]==fam]
            assert len(observations)==48
            assert approx(statistics.fmean(observations),value)
    comparisons={"nonlinear_vs_linear":("nonlinear","linear"),
                 "linear_vs_shuffled":("linear","shuffled_linear"),
                 "nonlinear_vs_mean":("nonlinear","mean")}
    for key,(first,second) in comparisons.items():
        gains=sum(r["standardized_reconstruction_MSE"][first]<
                  r["standardized_reconstruction_MSE"][second] for r in nonlate)
        losses=sum(r["standardized_reconstruction_MSE"][first]>
                   r["standardized_reconstruction_MSE"][second] for r in nonlate)
        actual=data["paired_primary_signs"][key]
        pinned=receipt["primary_paired_signs"][key]
        assert gains==actual["first_lower_error"]==pinned["first_lower_error"]
        assert losses==actual["second_lower_error"]==pinned["second_lower_error"]
        assert actual["ties"]==240-gains-losses
        assert approx(sign_p(gains,losses),actual["unadjusted_two_sided_sign_p"])
        assert approx(sign_p(gains,losses),pinned["unadjusted_two_sided_sign_p"])
    witnesses=data["nearest_endpoint_pairs"]
    assert len(witnesses)==240
    assert {r["a"] for r in witnesses}==set(xmap)
    for w in witnesses:
        a=xmap[w["a"]];b=xmap[w["b"]]
        assert a is not b
        d9=math.dist(a["h9"],b["h9"])
        d4=math.dist(a["h4"],b["h4"])
        assert approx(d9,w["endpoint_distance"])
        assert approx(d4,w["early_distance"])
        assert approx(d4/max(1.e-12,d9),w["early_to_endpoint_distance_ratio"])
        assert (a["family"]==b["family"])==w["same_source_family"]
    singular=data["local_jacobian_spectra"]
    assert len(singular)==receipt["jacobian_cases"]==24
    for item in singular:
        nums=item["singular_values"]
        assert len(nums)==12
        assert nums==sorted(nums)
        assert all(math.isfinite(s) and s>=0 for s in nums)
        assert approx(item["sigma_min"],nums[0])
        assert approx(item["sigma_max"],nums[-1])
        assert approx(item["condition_estimate"],nums[-1]/max(1.e-15,nums[0]))
    assert approx(min(z["sigma_min"] for z in singular),
                  receipt["min_estimated_singular_value"])
    assert approx(statistics.median(z["sigma_min"] for z in singular),
                  receipt["median_estimated_min_singular_value"])
    xstat=data["train_only_stats"]["endpoint"]
    ystat=data["train_only_stats"]["step4"]
    assert len(xstat["mean"])==len(ystat["mean"])==12
    assert len(data["train_only_stats"]["model_weights"]["linear"])==12
    assert len(data["train_only_stats"]["model_weights"]["nonlinear"])==12
    assert all(len(w)==37 for w in data["train_only_stats"]["model_weights"]["nonlinear"])
    return {"observed_test_cases":288,"unique_nonlate_inputs":240,
            "reconstruction_errors_recounted":len(allrecords)*5,
            "paired_sign_contrasts_recounted":3,"jacobian_spectra_audited":24}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--full",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    receipt=frozen_receipt()
    if args.full:
        raw=args.full.read_bytes()
        assert len(raw)==receipt["first_full_output_uncompressed_bytes"]
        assert sha256(raw).hexdigest()==SHA
        result=validate_full(json.loads(raw),receipt)
        print("NOR15A INDEPENDENT FULL RECEIPT AUDIT PASSED",result)
    else:
        print("NOR15A FIRST HOSTED RECEIPT HASH VALID")
    if args.check:
        assert receipt["original_first_green_run"].startswith("https://github.com/")


if __name__=="__main__":
    main()
