"""NOR-16 independent arithmetic and original-evidence provenance audit.

This module imports NO generator, GRU, policy, model or task grader.
It checks the original source-locked JSON bytes, every recorded decision,
candidate class, paid verification trace, public query simulation charge,
accepted-prediction posterior arithmetic, counterfactual candidate pair,
four cohort manifests and paired exact case comparisons.

The archived future target is intentionally NOT included in the source
result. Numerical ground truth entered only through each sealed original
grade. The independent full run must be reproduced separately in CI.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from fractions import Fraction
from hashlib import sha256
from math import comb
import json
from pathlib import Path
import random
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor16_first_frozen_receipt.json"
FULL_SHA="b1ca49c9fb0d95510f020d4318d29a10abc25759fa110370eef967d538a20cef"
SOURCE_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
ARMS=("original_4","consensus_4","hash_4","full_mask_original_16",
      "full_mask_map_16")
LATE="late_source"


def sign_p(gains,losses):
    n=gains+losses
    if n==0:
        return 1.
    return min(1.,2*sum(comb(n,k)
        for k in range(min(gains,losses)+1))/2**n)


def frozen():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor16.first_full_source_locked_public_query_receipt.v1"
    assert r["full_json_sha256"]==FULL_SHA
    assert r["full_json_bytes"]==1331006
    assert r["checkpoint_sha256"]==SOURCE_SHA
    assert r["late_primary_cases"]==192
    assert r["total_cases"]==288
    assert r["fresh_offset"]==93116000 and r["repetitions"]==4
    assert r["late_original_model_feature_groups"]==1
    assert r["signed_secondary_cases"]==r["rational_secondary_cases"]==48
    assert r["public_demo_degree"]==192
    assert r["public_query_input_degree"]==256
    assert r["four_check_budget"]==4
    assert r["full_mask_checks_per_case"]==16
    assert r["candidate_query_simulations_per_advanced_arm"]==16
    assert r["cached_source_unary_ops_per_advanced_arm"]==4
    assert set(r["arms"])==set(ARMS)
    assert r["arms"]["original_4"]==[79,99,778,33,99,466]
    assert r["arms"]["consensus_4"]==[82,96,777,36,96,465]
    assert r["arms"]["full_mask_map_16"]==[176,112,4608,80,112,3072]
    assert r["late_counterfactual_observation_collision_cases"]==140
    assert r["late_full_public_exact_mask_distinct"]==27
    assert r["late_all16_accept_cases"]==59
    assert r["uniform_candidate_prior_aggregate_conditional_expected_MAP_success"]==78.9375
    assert not r["four_check_consensus_primary_pass"]
    return r


def verify_one_record(r):
    """Independently rederive all public-mask, choice and cost arithmetic."""
    assert set(r)=={
        "model_input_sha256","public_demos_sha256","full_public_input_sha256",
        "public_query_input_sha256","family","public_accept_mask",
        "accepted_candidates","candidate_public_prediction_sha256",
        "query_output_equivalence_group_sizes",
        "accepted_candidate_posterior_group_count",
        "uniform_accepted_candidate_MAP_future_probability_num",
        "uniform_accepted_candidate_MAP_future_probability_den",
        "max_prediction_group_mass_all_classes",
        "model_original_top_four","consensus_top_four","hash_top_four",
        "counterfactual_indistinguishable_pair","arms",
    }
    for key in ("model_input_sha256","public_demos_sha256",
                "full_public_input_sha256","public_query_input_sha256"):
        assert len(r[key])==64
    mask=r["public_accept_mask"]
    assert 1<=mask<=65535
    assert r["accepted_candidates"]==mask.bit_count()
    digests=r["candidate_public_prediction_sha256"]
    assert len(digests)==16 and all(len(x)==64 for x in digests)
    classes=Counter(digests)
    groups=sorted(classes.values(),reverse=True)
    assert groups==r["query_output_equivalence_group_sizes"]
    assert r["max_prediction_group_mass_all_classes"]==groups[0]
    accepted=Counter(digests[j] for j in range(16) if mask&(1<<j))
    assert r["accepted_candidate_posterior_group_count"]==len(accepted)
    assert r["uniform_accepted_candidate_MAP_future_probability_den"]==mask.bit_count()
    assert r["uniform_accepted_candidate_MAP_future_probability_num"]==max(accepted.values())
    original=r["model_original_top_four"]
    consensus=r["consensus_top_four"]
    hashed=r["hash_top_four"]
    assert all(len(x)==4 and len(set(x))==4 for x in (original,consensus,hashed))
    assert all(0<=j<16 for x in (original,consensus,hashed) for j in x)
    # The negative-control shuffle is fully reproducible from public QUERY INPUT hash.
    random_ids=list(range(16))
    random.Random(int(r["public_query_input_sha256"][:16],16)).shuffle(random_ids)
    assert hashed==random_ids[:4]
    # Query-consensus top-four may break ties according to the full hidden
    # rank, but candidate agreement cluster sizes must be nonincreasing.
    order_mass=[classes[digests[j]] for j in consensus]
    assert order_mass==sorted(order_mass,reverse=True)
    pair=r["counterfactual_indistinguishable_pair"]
    if pair is not None:
        assert set(pair)=={
            "candidate_classes","first_divergent_future_output_degree",
            "publicly_accepted_both","same_complete_public_demos_and_query_input",
            "candidate_outputs_different"}
        a,b=pair["candidate_classes"]
        assert a!=b and mask&(1<<a) and mask&(1<<b)
        assert digests[a]!=digests[b]
        assert 0<=pair["first_divergent_future_output_degree"]<=256
        assert pair["publicly_accepted_both"]
        assert pair["same_complete_public_demos_and_query_input"]
        assert pair["candidate_outputs_different"]
    assert set(r["arms"])==set(ARMS)
    for name,arm in r["arms"].items():
        assert set(arm)=={
            "chosen_class","verification_trace","public_checks",
            "query_prediction_candidates","public_unary_query_transforms",
            "grade"}
        assert arm["grade"].keys()=={"accepted","future_correct","false_accept"}
        grade=arm["grade"]
        assert all(grade[k] in (0,1) for k in grade)
        assert grade["accepted"]==grade["future_correct"]+grade["false_accept"]
        chosen=arm["chosen_class"]
        assert chosen is None or (0<=chosen<16 and mask&(1<<chosen))
        if chosen is None:
            assert grade["accepted"]==0
        else:
            assert grade["accepted"]==1
        trace=arm["verification_trace"]
        if name.endswith("_4"):
            assert 1<=len(trace)<=4
            assert arm["public_checks"]==len(trace)
            assert len(trace)==len(set(trace))
            assert all(not mask&(1<<j) for j in trace[:-1])
            assert chosen==(trace[-1] if mask&(1<<trace[-1]) else None)
            if chosen is None:
                assert len(trace)==4
            expected={"original_4":original,"consensus_4":consensus,
                      "hash_4":hashed}[name]
            assert trace==expected[:len(trace)]
        else:
            assert trace==[]
            assert chosen is not None
            assert arm["public_checks"]==16
        if name=="original_4":
            assert arm["query_prediction_candidates"]==0
            assert arm["public_unary_query_transforms"]==0
        else:
            assert arm["query_prediction_candidates"]==16
            assert arm["public_unary_query_transforms"]==4
    # For equal-check conditional MAP, chosen candidate is in an accepted
    # class with the highest number of posterior-consistent programs.
    selected=r["arms"]["full_mask_map_16"]["chosen_class"]
    assert accepted[digests[selected]]==max(accepted.values())
    # If original four-check policy accepted, sixteen-check continuation
    # of the same frozen order must choose exactly the same candidate.
    orig_selected=r["arms"]["original_4"]["chosen_class"]
    if orig_selected is not None:
        assert r["arms"]["full_mask_original_16"]["chosen_class"]==orig_selected
    return pair is not None


def paired(records,left,right):
    matched=[r for r in records if r["family"]==LATE]
    gains=losses=0
    for r in matched:
        a=r["arms"][left]["grade"]["future_correct"]
        b=r["arms"][right]["grade"]["future_correct"]
        gains+=b>a
        losses+=a>b
    return {
        "family":LATE,
        "net":gains-losses,
        "paired_gains":gains,
        "paired_losses":losses,
        "primary_distinct_public_inputs":len(matched),
        "two_sided_exact_case_sign_p":sign_p(gains,losses)
    }


def inspect_full(data,pinned):
    assert data["schema"]=="nor16.public_query_predictive_consensus_source_locked.v1"
    assert data["source_checkpoint_sha256"]==SOURCE_SHA
    assert data["ninety_six_original_source_train_manifest"]==pinned["train_manifest"]
    assert data["thirty_six_original_source_dev_manifest"]==pinned["dev_manifest"]
    assert data["fresh_seed_offset"]==93116000
    assert data["fresh_repetitions"]==4
    assert data["primary_late_per_repetition"]==48
    assert data["secondary_signed_per_repetition"]==12
    assert data["secondary_rational_per_repetition"]==12
    assert data["all_cases"]==288
    assert data["late_source_primary_public_distinct_inputs"]==192
    assert data["late_source_unique_gru_visible_input_tensors"]==1
    assert data["nonlate_secondary_distinct_gru_inputs"]==96
    assert data["public_query_input_degree"]==256
    assert data["public_demo_observation_degree"]==192
    assert data["original_native_four_check_budget"]==4
    assert data["four_public_query_unary_ops_cached_per_case"]==4
    assert data["extra_full_mask_public_checks_per_case"]==16
    assert data["primary_Bonferroni_two_control_threshold"]==.025
    assert not data["consensus_four_passed_two_control_requirement"]
    scope=data["scope"]
    assert scope["public_demo_coefficients_only_for_verifier"]
    assert scope["query_input_degree_256_is_public"]
    assert scope["withheld_target_output_only_seen_by_sealed_grader"]
    assert scope["candidate_program_identity_never_read_in_policy"]
    assert scope["query_simulation_costs_reported_separately"]
    assert scope["full_mask_sixteen_checks_not_equal_cost_to_native_four"]
    assert scope["old_gru_weights_and_original_ranker_frozen"]
    assert scope["uniform_program_posterior_requires_uniform_prior_assumption"]
    assert scope["counterfactual_pairs_are_synthetic_possible_futures"]
    assert scope["no_retrospective_oracle_tuning_for_test"]
    records=data["records"]
    assert len(records)==288
    families=Counter(r["family"] for r in records)
    assert families=={"late_source":192,"signed":48,"rational":48}
    assert len({r["full_public_input_sha256"] for r in records})==288
    assert len({r["public_demos_sha256"] for r in records})==288
    late=[r for r in records if r["family"]==LATE]
    nonlate=[r for r in records if r["family"]!=LATE]
    assert len({r["model_input_sha256"] for r in late})==1
    assert len({r["model_input_sha256"] for r in nonlate})==96
    assert {r["model_input_sha256"] for r in late}.isdisjoint({
        r["model_input_sha256"] for r in nonlate})
    certified=sum(verify_one_record(r) for r in records)
    assert certified==(data["late_source_counterfactual_observation_collision_witnesses"]
        + sum(r["counterfactual_indistinguishable_pair"] is not None
              for r in nonlate))
    assert sum(r["counterfactual_indistinguishable_pair"] is not None
               for r in late)==pinned["late_counterfactual_observation_collision_cases"]==140
    assert len({r["public_accept_mask"] for r in late})==27
    assert sum(r["public_accept_mask"]==65535 for r in late)==59
    posterior=sum((Fraction(
        r["uniform_accepted_candidate_MAP_future_probability_num"],
        r["uniform_accepted_candidate_MAP_future_probability_den"])
        for r in late),Fraction(0))
    assert posterior==Fraction(1263,16)
    assert float(posterior)==pinned[
        "uniform_candidate_prior_aggregate_conditional_expected_MAP_success"]
    assert data["late_source_uniform_posterior_predictive_MAP_expected_success"]==float(posterior)
    assert data["late_source_exact_public_masks"]==27
    assert data["late_source_all_sixteen_public_accept_rate"]==59/192

    for rep,c in enumerate(data["fresh_cohorts"]):
        assert c["replicate"]==rep and c["cases"]==72
        block=records[rep*72:(rep+1)*72]
        assert Counter(x["family"] for x in block)=={
            "late_source":48,"signed":12,"rational":12}
        sha_demos=sha256("\n".join(
            x["public_demos_sha256"] for x in block).encode()).hexdigest()
        sha_public=sha256("\n".join(
            x["full_public_input_sha256"] for x in block).encode()).hexdigest()
        assert sha_demos==c["public_demo_manifest_sha256"]==pinned[
            "cohort_public_demo_manifests"][rep]
        assert sha_public==c["full_public_manifest_sha256"]==pinned[
            "cohort_full_public_manifests"][rep]
        for name,field in (("original_4","original_future_correct"),
                           ("consensus_4","consensus_four_future_correct"),
                           ("full_mask_map_16","full_mask_map_future_correct")):
            assert c[field]==sum(x["arms"][name]["grade"]["future_correct"]
                                  for x in block)
    for name,arm in data["arms"].items():
        assert name in ARMS
        assert arm["cases"]==288
        cases=[r["arms"][name] for r in records]
        actual=[
            sum(x["grade"]["future_correct"] for x in cases),
            sum(x["grade"]["false_accept"] for x in cases),
            sum(x["public_checks"] for x in cases),
            sum(r["arms"][name]["grade"]["future_correct"] for r in late),
            sum(r["arms"][name]["grade"]["false_accept"] for r in late),
            sum(r["arms"][name]["public_checks"] for r in late),
        ]
        assert actual==pinned["arms"][name],(name,actual)
        assert actual[:3]==[arm["future_correct"],arm["false_accept"],
                            arm["public_checks"]]
        assert arm["public_query_prediction_candidates"]==sum(
            x["query_prediction_candidates"] for x in cases)
        assert arm["public_unary_query_transforms"]==sum(
            x["public_unary_query_transforms"] for x in cases)
        for family,fam in arm["by_family"].items():
            sub=[r["arms"][name] for r in records if r["family"]==family]
            assert fam=={
                "cases":len(sub),
                "future_correct":sum(z["grade"]["future_correct"] for z in sub),
                "false_accept":sum(z["grade"]["false_accept"] for z in sub),
                "public_checks":sum(z["public_checks"] for z in sub),
            }
    tests={
        "consensus_vs_original":("original_4","consensus_4"),
        "consensus_vs_hash":("hash_4","consensus_4"),
        "full_mask_map_vs_full_mask_original":(
            "full_mask_original_16","full_mask_map_16"),
        "full_mask_map_vs_original":("original_4","full_mask_map_16"),
    }
    assert set(data["predeclared_primary_comparisons"])==set(tests)
    for label,(a,b) in tests.items():
        observed=paired(records,a,b)
        published=data["predeclared_primary_comparisons"][label]
        pinned_comp=pinned["primary_paired"][label]
        for key,value in observed.items():
            if key=="two_sided_exact_case_sign_p":
                assert abs(value-published[key])<1e-14
                assert abs(value-pinned_comp["p"])<1e-14
            else:
                assert value==published[key]
        assert observed["paired_gains"]==pinned_comp["gains"]
        assert observed["paired_losses"]==pinned_comp["losses"]
    return {"case_count":288,
            "distinct_primary_complete_public_sources":192,
            "native_and_public_check_records":len(records)*len(ARMS),
            "counterfactual_input_collision_witnesses":140,
            "exact_candidate_prediction_digest_entries":len(records)*16,
            "four_paired_contrasts_recomputed":4,
            "bayesian_MAP_expected_late_count":float(posterior),
            "four_check_consensus_primary_pass":False}


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group()
    group.add_argument("--full",type=Path)
    group.add_argument("--archive",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    pinned=frozen()
    raw=None
    if args.full:
        raw=args.full.read_bytes()
    if args.archive:
        with ZipFile(args.archive) as zipfile:
            assert zipfile.namelist()==["nor16_full.json"]
            assert zipfile.testzip() is None
            raw=zipfile.read("nor16_full.json")
    if raw is not None:
        assert len(raw)==pinned["full_json_bytes"]
        assert sha256(raw).hexdigest()==FULL_SHA
        result=inspect_full(json.loads(raw),pinned)
        print("NOR16 INDEPENDENT PUBLIC QUERY DECISION AUDIT PASSED",result)
    else:
        print("NOR16 IMMUTABLE FIRST RUN RECEIPT VALID")
    if args.check:
        assert pinned["first_green_run"].startswith("https://github.com/")


if __name__=="__main__":
    main()
