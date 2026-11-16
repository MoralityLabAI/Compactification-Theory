"""NOR-18 independent source-locked coefficient-work and certificate auditor.

No experiment, model, generator, or verifier imports. Reconstruct every
fixed arithmetic-work price from the public header and every charged policy
trace from immutable original results. Verify that no full public-accepted
candidate is removed by a prefix, all certifications are sound for the
recorded candidate universe, and each arm respects the same total cap.

The complete native verification and sealed future grader are rerun by
the companion hosted experiment. This auditor checks its *records* and
their immutable byte fingerprint rather than re-executing its source code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from math import comb
import json
from pathlib import Path
import random
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor18_first_frozen_receipt.json"
FULL_SHA="9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b"
CK_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
DEGREES=(112,128,144,160,176,192)
BUDGETS=(4,8,16)
OPS=("I","N","R","S")


def unary(op,degree):
    assert op in OPS and degree>=0
    return degree*(degree+1)//2 if op=="R" else degree+1


def combination(header,degree,equality):
    k=header["delay"];delta=header["spacing"]
    assert degree>=k and k in (64,80,96,112) and delta in (24,32)
    return (degree+1)+1+2*(
        max(0,degree-k-delta+1)+
        max(0,degree-k-2*delta+1))+(degree+1 if equality else 0)


def check_cost(header,j,degree=192):
    assert 0<=j<16
    return 2*(unary(header["base"],degree)+
              unary(OPS[j//4],degree)+unary(OPS[j%4],degree)+
              combination(header,degree,True))


def screen_cost(header,degree):
    return 2*(sum(unary(op,degree) for op in OPS)
              +16*combination(header,degree,True))


def query_cost(header):
    return (sum(unary(op,256) for op in OPS)
            +16*combination(header,256,False))


def work_costs(header):
    costs=[check_cost(header,j) for j in range(16)]
    return {
        "full_check_costs":costs,
        "screen_setup_costs":{str(d):screen_cost(header,d) for d in DEGREES},
        "query_simulation_cost":query_cost(header),
        "public_full_check_unit":(sum(costs)+15)//16,
        "reference_unit_uses_public_header_only":True,
        "work_formula_version":"nor18.coefficient_operation_proxy.v1",
    }


def sign_p(gains,losses):
    n=gains+losses
    if n==0:return 1.
    return min(1.,2*sum(comb(n,k) for k in range(min(gains,losses)+1))/2**n)


def pinned():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor18.frozen_coefficient_work_depth_frontier_first_hosted_receipt.v1"
    assert r["original_full_json_sha256"]==FULL_SHA
    assert r["original_full_json_uncompressed_bytes"]==12896423
    assert r["frozen_gru_checkpoint_sha256"]==CK_SHA
    assert r["new_seed_offset"]==93518000
    assert r["source_cases_total"]==288
    assert r["primary_late_source_cases"]==192
    assert r["nonlate_model_visible_unique_inputs"]==96
    assert r["prefix_degrees"]==list(DEGREES[:-1])
    assert r["batched_full_degree"]==192
    assert r["work_multipliers"]==list(BUDGETS)
    assert r["policy_arms"]==28
    assert len(r["new_cohort_public_manifests"])==4
    assert r["primary_full_future_output_certifiable"]==48
    assert r["preregistered_primary_passed"]
    assert r["arm_certified_false_accepts"]==0
    assert r["primary_late_source_certificates"]["d176_4"]==8
    assert r["primary_late_source_certificates"]["batch192_8"]==48
    return r


def expected_name(name):
    level,suffix=name.rsplit("_",1)
    budget=int(suffix)
    assert budget in BUDGETS
    if level=="greedy":
        return "greedy",None,budget
    if level=="none":
        return "consensus",None,budget
    if level=="batch192":
        return "batch",192,budget
    if level in ("d160_original","d176_original"):
        return "original",int(level[1:4]),budget
    if level in ("d160_hash","d176_hash"):
        return "hash",int(level[1:4]),budget
    assert level.startswith("d") and int(level[1:]) in DEGREES[:-1]
    return "consensus",int(level[1:]),budget


def check_record(row,expected_arms):
    assert row["family"] in ("late_source","signed","rational")
    for key in ("public_demos_sha256","full_public_input_sha256",
                "model_visible_sha256","public_query_input_sha256"):
        assert len(row[key])==64
    header=row["header"]
    assert set(header)=={"base","delay","spacing","amplitude"}
    assert header["base"] in OPS
    assert header["amplitude"] in (-2,-1,1,2)
    assert row["cost_register"]==work_costs(header)
    costs=row["cost_register"]
    cost_unit=costs["public_full_check_unit"]
    full=row["retro_full_accepted_mask"]
    masks=row["public_prefix_survivor_masks"]
    assert set(masks)=={str(d) for d in DEGREES}
    assert 0<full<65536 and full==masks["192"]
    assert all(0<mask<65536 for mask in masks.values())
    assert all(masks[str(b)]&masks[str(a)]==masks[str(b)]
               for a,b in zip(DEGREES,DEGREES[1:]))
    assert row["retro_public_candidate_checks"]==16
    assert len(row["public_query_output_prediction_sha256"])==16
    predictions=row["public_query_output_prediction_sha256"]
    assert all(len(p)==64 for p in predictions)
    future_sig={predictions[j] for j in range(16) if full&(1<<j)}
    assert row["certificate_possible_from_complete_public_mask"]==(
        len(future_sig)==1)
    assert set(row["arms"])==expected_arms
    def output_multiplicity(mask):
        return Counter(predictions[j] for j in range(16)
                       if mask&(1<<j))
    original_top_four=None
    for name,arm in row["arms"].items():
        kind,degree,budget=expected_name(name)
        assert arm["work_multiplier"]==budget
        assert arm["reference_full_check_unit"]==cost_unit
        limit=budget*cost_unit
        assert arm["work_cap"]==limit
        assert arm["ranking"]==kind if kind!="greedy" else arm["ranking"]=="original"
        assert arm["screen_degree"]==degree
        assert 0<=arm["total_work"]<=limit
        assert arm["total_work"]==(arm["query_work"]+
            arm["screen_work"]+arm["full_candidate_check_work"])
        assert arm["full_candidate_checks"]==len(arm["trace"])
        candidate_query=0 if kind=="greedy" else costs["query_simulation_cost"]
        screen=costs["screen_setup_costs"][str(degree)] if degree is not None else 0
        setup=candidate_query+screen
        feasible=setup<=limit
        assert arm["setup_feasible"]==feasible
        if not feasible:
            assert arm["total_work"]==0 and arm["trace"]==[]
            assert arm["exit"]=="setup_over_work_cap"
            assert arm["chosen_class"] is None and not arm["certified"]
            assert arm["remaining_viable_mask"] is None
            assert arm["initial_viable_mask"]==65535
            assert arm["query_work"]==arm["screen_work"]==0
            assert arm["screen_public_comparisons"]==0
            assert arm["batched_degree192_public_comparisons"]==0
            assert arm["query_public_candidate_outputs"]==0
            assert arm["query_public_unary_maps"]==0
            continue
        assert arm["query_work"]==candidate_query
        assert arm["screen_work"]==screen
        assert arm["query_public_candidate_outputs"]==(0 if kind=="greedy" else 16)
        assert arm["query_public_unary_maps"]==(0 if kind=="greedy" else 4)
        assert arm["screen_public_comparisons"]==(16 if degree is not None else 0)
        assert arm["screen_public_demo_compositions"]==(32 if degree is not None else 0)
        assert arm["screen_public_unary_maps"]==(8 if degree is not None else 0)
        assert arm["batched_degree192_public_comparisons"]==(16 if kind=="batch" else 0)
        mask=65535 if degree is None else masks[str(degree)]
        assert arm["initial_viable_mask"]==mask
        assert full&mask==full
        grade=arm["grade"]
        assert set(grade)=={"accepted","future_correct","false_accept"}
        assert grade["accepted"]==grade["future_correct"]+grade["false_accept"]
        assert all(grade[k] in (0,1) for k in grade)
        chosen=arm["chosen_class"]
        if chosen is None:
            assert grade=={"accepted":0,"future_correct":0,"false_accept":0}
        else:
            assert 0<=chosen<16 and full&(1<<chosen)
            assert grade["accepted"]==1
        if kind=="batch":
            assert arm["trace"]==[]
            assert arm["full_candidate_check_work"]==0
            assert arm["remaining_viable_mask"]==mask
            if arm["certified"]:
                assert chosen is not None and len(future_sig)==1
                assert predictions[chosen] in future_sig
                assert arm["exit"]=="certified_batched_full_public"
                assert grade["future_correct"]==1 and grade["false_accept"]==0
            else:
                assert chosen is None
                assert arm["exit"]=="ambiguous_batched_full_public"
            continue
        trace=arm["trace"]
        assert len({j["class"] for j in trace})==len(trace)
        for entry in trace:
            j=entry["class"]
            assert 0<=j<16 and mask&(1<<j)
            assert type(entry["accepted"]) is bool
            assert entry["accepted"]==bool(full&(1<<j))
            assert entry["charged_work"]==costs["full_check_costs"][j]
        assert arm["full_candidate_check_work"]==sum(
            j["charged_work"] for j in trace)
        assert arm["total_work"]==setup+arm["full_candidate_check_work"]
        positives=[j["class"] for j in trace if j["accepted"]]
        if kind=="greedy":
            assert arm["remaining_viable_mask"] is None
            assert not arm["certified"]
            assert len(positives)<=1
            if chosen is not None:
                assert chosen==positives[0]==trace[-1]["class"]
                assert arm["exit"]=="first_public_accept"
            else:
                assert positives==[]
                assert arm["exit"] in ("next_check_over_work_cap","all_viable_checked")
            continue
        rejected=sum(1<<j["class"] for j in trace if not j["accepted"])
        remaining=mask&~rejected
        assert arm["remaining_viable_mask"]==remaining
        assert full&remaining==full
        remaining_future={predictions[j] for j in range(16) if remaining&(1<<j)}
        if arm["certified"]:
            assert chosen is not None and chosen in positives
            assert len(remaining_future)==1
            assert arm["exit"]=="certified_unanimous_remaining"
            assert grade["future_correct"]==1 and grade["false_accept"]==0
            assert len(future_sig)==1 and predictions[chosen] in future_sig
        else:
            assert chosen is None
            if len({predictions[j] for j in positives})>1:
                assert arm["exit"]=="incompatible_full_accepted_outputs"
            else:
                assert arm["exit"] in ("next_check_over_work_cap",
                                     "all_viable_checked","no_viable_candidates")
        masses=output_multiplicity(mask)
        if kind=="consensus":
            values=[masses[predictions[j["class"]]] for j in trace]
            assert values==sorted(values,reverse=True)
        if kind=="hash":
            ids=list(range(16))
            rng=random.Random(int(row["public_query_input_sha256"][:16],16))
            rng.shuffle(ids)
            expected=[j for j in ids if mask&(1<<j)]
            assert [entry["class"] for entry in trace]==expected[:len(trace)]
    return True


def aggregate(records):
    return {
        "cases":len(records),
        "returned_candidates":sum(r["chosen_class"] is not None for r in records),
        "certificates":sum(r["certified"] for r in records),
        "future_correct":sum(r["grade"]["future_correct"] for r in records),
        "false_accept":sum(r["grade"]["false_accept"] for r in records),
        "total_work":sum(r["total_work"] for r in records),
        "query_work":sum(r["query_work"] for r in records),
        "screen_work":sum(r["screen_work"] for r in records),
        "full_check_work":sum(r["full_candidate_check_work"] for r in records),
        "full_check_calls":sum(r["full_candidate_checks"] for r in records),
        "batch_full_public_comparisons":sum(
            r["batched_degree192_public_comparisons"] for r in records),
        "public_query_candidates":sum(
            r["query_public_candidate_outputs"] for r in records),
        "screen_public_comparisons":sum(
            r["screen_public_comparisons"] for r in records),
        "work_cap":sum(r["work_cap"] for r in records),
        "setup_feasible":sum(r["setup_feasible"] for r in records),
        "exit_reasons":dict(Counter(r["exit"] for r in records)),
    }


def paired(rows,left,right):
    primary=[r for r in rows if r["family"]=="late_source"]
    gains=sum(r["arms"][right]["certified"]>r["arms"][left]["certified"]
              for r in primary)
    losses=sum(r["arms"][right]["certified"]<r["arms"][left]["certified"]
               for r in primary)
    return {
        "cases":len(primary),"family":"late_source",
        "gains":gains,"losses":losses,"net":gains-losses,
        "comparison_measure":"certified",
        "exact_unadjusted_p":sign_p(gains,losses),
    }


def audit(data,receipt):
    assert data["schema"]=="nor18.common_work_cached_prefix_certificate_frontier.v1"
    assert data["source_checkpoint_sha256"]==CK_SHA
    assert data["frozen_train_manifest"]==receipt["frozen_original_train_manifest"]
    assert data["frozen_dev_manifest"]==receipt["frozen_original_dev_manifest"]
    assert data["seed_offset"]==93518000
    assert data["repetitions"]==4
    assert data["per_rep_late"]==48 and data["per_rep_signed"]==12
    assert data["per_rep_rational"]==12
    assert data["frozen_prefix_degrees"]==list(DEGREES[:-1])
    assert data["batched_full_degree"]==192
    assert data["frozen_work_multipliers"]==[4,8,16]
    assert data["work_model"]==receipt["work_model"]
    assert data["all_cases"]==288
    assert data["primary_late_distinct_complete_public_inputs"]==192
    assert data["primary_late_unique_model_feature_tensors"]==1
    assert data["nonlate_unique_model_feature_tensors"]==96
    assert data["retrospective_full_public_check_calls"]==4608
    assert len(data["arm_names"])==receipt["policy_arms"]==28
    assert data["primary_full_mask_output_certifiable"]==48
    assert data["preregistered_primary_two_control_passed"]
    assert data["primary_alpha"]==.025
    for flag in ("public_demo_prefix_only_for_screen",
        "query_simulated_from_public_input_only",
        "private_future_sealed_until_all_decisions_frozen",
        "original_1000_param_gru_immutable",
        "all_work_caps_apply_to_total_query_screen_and_full_check",
        "full_batch192_charges_16_full_public_candidate_comparisons",
        "prefix_candidates_never_exclude_full_accepted_member",
        "no_hidden_program_identity_in_any_policy",
        "proxy_excludes_bigint_bit_complexity_python_runtime_and_memory",
        "control_work_is_charged_independently_even_if_runner_caches",
        "retro_full_mask_checked_only_after_all_choices",
        "synthetic_generator_family_conditional_soundness_only"):
        assert data["scope"][flag],flag
    rows=data["records"]
    assert len(rows)==288
    assert Counter(r["family"] for r in rows)=={
        "late_source":192,"signed":48,"rational":48}
    assert len({r["full_public_input_sha256"] for r in rows})==288
    assert len({r["public_demos_sha256"] for r in rows})==288
    late=[r for r in rows if r["family"]=="late_source"]
    nonlate=[r for r in rows if r["family"]!="late_source"]
    assert len({r["model_visible_sha256"] for r in late})==1
    assert len({r["model_visible_sha256"] for r in nonlate})==96
    assert set(data["arm_names"])==set(receipt["primary_late_source_certificates"])
    assert all(check_record(r,set(data["arm_names"])) for r in rows)
    assert sum(r["certificate_possible_from_complete_public_mask"] for r in late)==48
    for index,cohort in enumerate(data["cohorts"]):
        assert cohort["replicate"]==index and cohort["cases"]==72
        batch=rows[index*72:(index+1)*72]
        assert Counter(r["family"] for r in batch)=={
            "late_source":48,"signed":12,"rational":12}
        digest=sha256("\n".join(r["full_public_input_sha256"] for r in batch).encode()).hexdigest()
        assert cohort["public_manifest_sha256"]==digest==receipt[
            "new_cohort_public_manifests"][index]
        for arm in data["arm_names"]:
            assert cohort["certificates_by_arm"][arm]==sum(
                r["arms"][arm]["certified"] for r in batch
                if r["family"]=="late_source")
    assert set(data["arms"])==set(data["arm_names"])
    for name,published in data["arms"].items():
        actual=aggregate([r["arms"][name] for r in rows])
        assert published["by_family"] and set(published["by_family"])=={
            "late_source","signed","rational"}
        for key,value in actual.items():
            assert published[key]==value,(name,key,value)
        primary=aggregate([r["arms"][name] for r in late])
        assert primary["certificates"]==receipt["primary_late_source_certificates"][name]
        assert published["by_family"]["late_source"]==primary
        for family in ("signed","rational"):
            a=aggregate([r["arms"][name] for r in rows if r["family"]==family])
            assert published["by_family"][family]==a
        if not name.startswith("greedy"):
            assert published["false_accept"]==0
    for name,val in receipt["primary_late_source_work_proxy"].items():
        assert data["arms"][name]["by_family"]["late_source"]["total_work"]==val
    for key,cap in receipt["primary_late_source_work_caps"].items():
        k=int(key)
        for name in data["arm_names"]:
            if name.endswith(f"_{k}"):
                assert data["arms"][name]["by_family"]["late_source"]["work_cap"]==cap
    for name,count in receipt["primary_setup_feasible_counts"].items():
        assert data["arms"][name]["by_family"]["late_source"]["setup_feasible"]==count
    comparisons={
        "d176_4_vs_d160_4":("d160_4","d176_4"),
        "d176_4_vs_none_4":("none_4","d176_4"),
        "d160_4_vs_none_4":("none_4","d160_4"),
        "batch192_4_vs_d176_4":("d176_4","batch192_4"),
        "batch192_8_vs_none_8":("none_8","batch192_8"),
        "d160_4_vs_d160_original_4":("d160_original_4","d160_4"),
        "d176_4_vs_d176_hash_4":("d176_hash_4","d176_4"),
    }
    assert set(data["paired_primary_and_secondary"])==set(comparisons)
    for name,(first,second) in comparisons.items():
        rebuilt=paired(rows,first,second)
        source=data["paired_primary_and_secondary"][name]
        rp=receipt["primary_pair_gains_losses_p"][name]
        assert rebuilt["gains"]==source["gains"]==rp[0]
        assert rebuilt["losses"]==source["losses"]==rp[1]
        assert rebuilt["net"]==source["net"]
        assert abs(rebuilt["exact_unadjusted_p"]-source["exact_unadjusted_p"])<1e-14
        assert abs(rebuilt["exact_unadjusted_p"]-rp[2])<1e-14
    for name,values in receipt["primary_four_cohort_certificates"].items():
        assert [c["certificates_by_arm"][name] for c in data["cohorts"]]==values
    # Exact cross-product sharing identity for EVERY source header and depth.
    for row in rows:
        h=row["header"]
        for degree in DEGREES:
            separate=sum(check_cost(h,j,degree) for j in range(16))
            cached=screen_cost(h,degree)
            four=sum(unary(op,degree) for op in OPS)
            assert separate-cached==2*(16*unary(h["base"],degree)+7*four)
    return {
        "fresh_source_cases":288,
        "independent_primary_complete_public_inputs":192,
        "policy_decisions_checked":288*28,
        "public_prediction_digests_checked":288*16,
        "exact_candidate_trace_costs_reconstructed":sum(
            len(arm["trace"]) for row in rows
            for arm in row["arms"].values()),
        "cached_unary_saving_identities_checked":288*6,
        "exact_paired_sign_contrasts_checked":7,
        "full_public_certifiable_cases":48,
        "preregistered_primary_gain":True,
        "all_certificate_false_accepts":0,
    }


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--full",type=Path)
    source.add_argument("--archive",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    receipt=pinned()
    raw=None
    if args.full:
        raw=args.full.read_bytes()
    if args.archive:
        with ZipFile(args.archive) as z:
            assert z.namelist()==["nor18_full.json"]
            assert z.testzip() is None
            raw=z.read("nor18_full.json")
    if raw is not None:
        assert len(raw)==receipt["original_full_json_uncompressed_bytes"]
        assert sha256(raw).hexdigest()==FULL_SHA
        report=audit(json.loads(raw),receipt)
        print("NOR18 INDEPENDENT COEFFICIENT-WORK AND CERTIFICATE AUDIT PASSED",report)
    else:
        print("NOR18 ORIGINAL HOSTED JSON SHA AND FIRST RESULT RECEIPT VALID")
    if args.check:
        assert receipt["original_hosted_run"].startswith("https://github.com/")


if __name__=="__main__":
    main()
