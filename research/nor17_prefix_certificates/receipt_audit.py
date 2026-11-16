"""NOR-17 independently verify frozen numerical receipt and all policy traces.

This file imports no GRU, task, screening, generator or policy code. It
checks original JSON byte identity, every policy's public checking ledger,
the soundness of short-prefix and full-rejection candidate elimination,
output-hash unanimity at every certified decision, source cohort manifests
and all 7 predeclared paired comparison counts. The original interpreter's
semantic correctness is checked separately by source-parity unit tests.

Supported --archive reads the source-controlled original full JSON ZIP.
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
RECEIPT=HERE/"results"/"nor17_first_frozen_receipt.json"
FULL_SHA="827b9aab53674567d7aa77c655e33c8800ba9a7372a0f3bdc889cdbf4d9c6b7b"
CK_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
ARMS=(
    "greedy_4","greedy_8","greedy_16",
    "unscreened_4","unscreened_8","unscreened_16",
    "prefix_4","prefix_8","prefix_16",
    "prefix_original_4","prefix_hash_4",
)


def exact_two_sided_sign_p(gain,loss):
    n=gain+loss
    if n==0:
        return 1.0
    return min(1.0,2*sum(comb(n,i) for i in range(min(gain,loss)+1))/2**n)


def pinned_receipt():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor17.frozen_source_locked_4_8_16_certificate_receipt.v1"
    assert r["original_full_json_sha256"]==FULL_SHA
    assert r["original_full_json_bytes"]==4174026
    assert r["frozen_checkpoint_sha256"]==CK_SHA
    assert r["fresh_seed_offset"]==93317000
    assert (r["fresh_repetitions"],r["all_source_cases"],
            r["primary_complete_public_late_sources"])==(4,288,192)
    assert r["paid_full_check_budgets"]==[4,8,16]
    assert (r["short_exact_prefix_degree"],r["full_public_demo_degree"],
            r["public_query_input_degree"])==(160,192,256)
    assert len(r["fresh_source_cohort_public_manifests"])==4
    assert len(set(r["fresh_source_cohort_public_manifests"]))==4
    assert set(r["arm_outcomes"])==set(ARMS)
    assert r["primary_late_full_future_identifiable_cases"]==47
    assert r["arm_outcomes"]["prefix_4"]==[28,28,0,619,124,124,0,820]
    assert r["arm_outcomes"]["prefix_16"]==[47,47,0,1119,143,143,0,1320]
    assert not r["registered_prefix4_two_control_success"]
    return r


def independent_record_check(row):
    assert row["family"] in ("late_source","signed","rational")
    for name in ("public_demos_sha256","full_public_input_sha256",
                 "model_input_sha256","public_query_input_sha256"):
        assert len(row[name])==64
    full=row["full_public_accept_mask_retrospective"]
    short=row["short_prefix_survivor_mask"]
    assert 0<full<65536 and 0<short<65536
    assert full&short==full
    assert row["initial_possible_generator_count"]==short.bit_count()
    assert row["original_public_accepted_count"]==full.bit_count()
    pred=row["hypothetical_query_output_hashes"]
    assert len(pred)==16 and all(len(h)==64 for h in pred)
    full_output_group={pred[j] for j in range(16) if full&(1<<j)}
    prefix_output_group={pred[j] for j in range(16) if short&(1<<j)}
    assert row["full_public_future_output_identifiable"]==(len(full_output_group)==1)
    assert row["short_prefix_future_output_identifiable"]==(len(prefix_output_group)==1)
    assert row["retrospective_public_full_checks"]==16
    assert len(row["original_top_four"])==4
    assert len(set(row["original_top_four"]))==4
    assert set(row["arms"])==set(ARMS)
    random_ids=list(range(16))
    random.Random(int(row["public_query_input_sha256"][:16],16)).shuffle(random_ids)
    for name,result in row["arms"].items():
        budget=int(name.rsplit("_",1)[-1])
        assert budget in (4,8,16)
        is_greedy=name.startswith("greedy")
        is_prefix=name.startswith("prefix")
        assert result["public_checks"]==len(result["trace"])
        assert 1<=len(result["trace"])<=budget
        assert len({j["class"] for j in result["trace"]})==len(result["trace"])
        assert set(result["grade"])=={"accepted","future_correct","false_accept"}
        grade=result["grade"]
        assert all(grade[k] in (0,1) for k in grade)
        assert grade["accepted"]==grade["future_correct"]+grade["false_accept"]
        tested=result["trace"]
        assert all(isinstance(t["class"],int) and 0<=t["class"]<16
                   and type(t["accepted"]) is bool for t in tested)
        assert all(bool(full&(1<<t["class"]))==t["accepted"] for t in tested)
        positives=[t["class"] for t in tested if t["accepted"]]
        failed_mask=sum(1<<t["class"] for t in tested if not t["accepted"])
        assert result["confirmed_accepted_classes"]==positives
        assert result["candidate_query_compositions"]==(
            0 if is_greedy else 16)
        assert result["query_unary_transformations"]==(
            0 if is_greedy else 4)
        assert result["prefix_checks"]==(16 if is_prefix else 0)
        assert result["prefix_demo_compositions"]==(
            32 if is_prefix else 0)
        assert result["prefix_unary_transformations"]==(
            8 if is_prefix else 0)
        initial=short if is_prefix else 65535
        assert result["initial_viable_mask"]==initial
        assert all(initial&(1<<t["class"]) for t in tested)
        if is_greedy:
            assert result["remaining_viable_mask"] is None
            assert not result["certified"]
            assert result["full_output_unanimity"] is None
            assert len(positives)<=1
            chosen=positives[0] if positives else None
            assert chosen==result["chosen_class"]
            if chosen is None:
                assert len(tested)==budget
                assert grade["accepted"]==0
                assert result["exit"]=="budget_exhausted"
            else:
                assert tested[-1]["class"]==chosen
                assert grade["accepted"]==1
                assert result["exit"]=="first_public_accept"
            if name=="greedy_4":
                assert [z["class"] for z in tested]==row["original_top_four"][:len(tested)]
            if len(tested)>=4:
                assert [z["class"] for z in tested[:4]]==row["original_top_four"]
            continue

        remaining=initial&~failed_mask
        assert remaining!=0
        assert result["remaining_viable_mask"]==remaining
        assert full&remaining==full
        possible_outputs={pred[j] for j in range(16) if remaining&(1<<j)}
        unanimous=(len(possible_outputs)==1)
        assert result["full_output_unanimity"]==unanimous
        chosen=result["chosen_class"]
        if result["certified"]:
            assert chosen is not None and chosen in positives
            assert unanimous
            assert result["exit"]=="complete_public_output_certificate"
            assert grade=={"accepted":1,"false_accept":0,"future_correct":1}
            assert len({pred[j] for j in range(16) if full&(1<<j)})==1
            assert pred[chosen] in possible_outputs
        else:
            assert chosen is None
            assert grade=={"accepted":0,"false_accept":0,"future_correct":0}
            if len({pred[j] for j in positives})>1:
                assert result["exit"]=="incompatible_full_accepted_outputs"
            else:
                assert result["exit"] in {
                    "budget_exhausted",
                    "budget_exhausted_unverified",
                    "no_full_accepted_candidate",
                }
        if name=="prefix_hash_4":
            expected=[i for i in random_ids if short&(1<<i)]
            assert [t["class"] for t in tested]==expected[:len(tested)]
        if name in {"prefix_4","prefix_8","prefix_16",
                    "unscreened_4","unscreened_8","unscreened_16"}:
            counts=Counter(pred[j] for j in range(16) if initial&(1<<j))
            masses=[counts[pred[t["class"]]] for t in tested]
            assert masses==sorted(masses,reverse=True)
    return 1


def independent_aggregate(rows):
    return {
        "cases":len(rows),
        "returned_proposals":sum(x["chosen_class"] is not None for x in rows),
        "certificates":sum(x["certified"] for x in rows),
        "future_correct":sum(x["grade"]["future_correct"] for x in rows),
        "false_accept":sum(x["grade"]["false_accept"] for x in rows),
        "full_public_checks":sum(x["public_checks"] for x in rows),
        "prefix_candidate_checks":sum(x["prefix_checks"] for x in rows),
        "prefix_demo_candidate_compositions":sum(
            x["prefix_demo_compositions"] for x in rows),
        "prefix_unary_maps":sum(x["prefix_unary_transformations"] for x in rows),
        "query_candidate_compositions":sum(
            x["candidate_query_compositions"] for x in rows),
        "query_unary_maps":sum(x["query_unary_transformations"] for x in rows),
        "by_exit":dict(Counter(x["exit"] for x in rows)),
    }


def paired_records(records,left,right,measure="certified"):
    rows=[r for r in records if r["family"]=="late_source"]
    def get(arm,record):
        return (record["arms"][arm]["certified"] if measure=="certified"
                else record["arms"][arm]["grade"]["future_correct"])
    gain=sum(get(right,r)>get(left,r) for r in rows)
    loss=sum(get(right,r)<get(left,r) for r in rows)
    return {"family":"late_source","cases":len(rows),
            "gains":gain,"losses":loss,"net":gain-loss,
            "exact_unadjusted_p":exact_two_sided_sign_p(gain,loss),
            "comparison_measure":measure}


def full_audit(data,pinned):
    assert data["schema"]=="nor17.risk_bounded_prefix_exclusion_public_output_certificate.v1"
    assert data["source_checkpoint_sha256"]==CK_SHA
    assert data["frozen_train_manifest"]==pinned["original_train_public_manifest_sha256"]
    assert data["frozen_dev_manifest"]==pinned["original_dev_public_manifest_sha256"]
    assert data["fresh_seed_offset"]==pinned["fresh_seed_offset"]
    assert data["repetitions"]==4
    assert data["late_source_per_repetition"]==48
    assert data["signed_per_repetition"]==12
    assert data["rational_per_repetition"]==12
    assert data["source_public_prefix_degree"]==160
    assert data["demo_public_degree"]==192
    assert data["public_query_input_degree"]==256
    assert data["policy_budgets"]==[4,8,16]
    assert data["arm_names"]==list(ARMS)
    assert data["all_cases"]==288
    assert data["primary_distinct_complete_public_late_cases"]==192
    assert data["primary_unique_gru_input_groups"]==1
    assert data["nonlate_distinct_gru_input_groups"]==96
    assert data["retrospective_full_source_audit_checks"]==4608
    assert data["preregistered_two_control_alpha"]==.025
    assert not data["prefix_rank4_passed_two_control_criterion"]
    scope=data["scope"]
    for flag in (
        "all_generator_members_in_enumerated_candidate_family",
        "certificates_require_one_exact_full_public_accepted_candidate",
        "every_non_rejected_public_candidate_must_predict_same_output",
        "prefix_only_elimination_is_exact_and_sound",
        "offline_full16_public_mask_audit_after_all_policy_choices",
        "offline_audit_cost_separate_from_inference",
        "private_future_used_only_by_sealed_grade_after_choice",
        "original_1000_param_gru_and_ranker_immutable",
        "model_visible_late_inputs_aliased_but_complete_public_distinct",
        "prefix_filter_unary_and_composition_costs_reported",
        "query_unary_and_composition_costs_reported",
        "no_general_theorem_prover_soundness_claim",
        "no_adjustment_to_heldout_policy_or_source_seed",
    ):
        assert scope[flag],flag
    records=data["records"]
    assert len(records)==288
    families=Counter(r["family"] for r in records)
    assert families=={"late_source":192,"signed":48,"rational":48}
    assert len({r["full_public_input_sha256"] for r in records})==288
    assert len({r["public_demos_sha256"] for r in records})==288
    late=[r for r in records if r["family"]=="late_source"]
    nonlate=[r for r in records if r["family"]!="late_source"]
    assert len({r["model_input_sha256"] for r in late})==1
    assert len({r["model_input_sha256"] for r in nonlate})==96
    assert {r["model_input_sha256"] for r in late}.isdisjoint({
        r["model_input_sha256"] for r in nonlate})
    assert sum(independent_record_check(r) for r in records)==288
    assert data["primary_full_public_future_output_identifiable"]==sum(
        r["full_public_future_output_identifiable"] for r in late)==47
    assert data["primary_short_prefix_future_output_identifiable"]==sum(
        r["short_prefix_future_output_identifiable"] for r in late)==0
    assert data["source_wide_certified_future_count"]==sum(
        r["short_prefix_future_output_identifiable"] for r in records)
    hist=Counter(r["initial_possible_generator_count"] for r in late)
    assert {str(k):v for k,v in sorted(hist.items())}==pinned[
        "primary_late_short_prefix_survivor_count_histogram"]
    assert len(data["cohorts"])==4
    for rep,c in enumerate(data["cohorts"]):
        assert c["replicate"]==rep and c["cases"]==72
        subset=records[rep*72:(rep+1)*72]
        assert Counter(r["family"] for r in subset)=={
            "late_source":48,"signed":12,"rational":12}
        manifest=sha256("\n".join(
            r["full_public_input_sha256"] for r in subset).encode()).hexdigest()
        assert c["public_manifest_sha256"]==manifest==pinned[
            "fresh_source_cohort_public_manifests"][rep]
        counts=[sum(r["arms"][f"prefix_{budget}"]["certified"]
                    for r in subset if r["family"]=="late_source")
                for budget in (4,8,16)]
        assert counts==pinned["primary_certificates_by_cohort_at_4_8_16"][rep]
        assert counts==[c["primary_certificate_by_budget"][str(b)]
                        for b in (4,8,16)]
        assert [sum(r["arms"][f"greedy_{b}"]["grade"]["future_correct"]
                    for r in subset if r["family"]=="late_source")
                for b in (4,8,16)]==[
                    c["primary_greedy_correct_by_budget"][str(b)]
                    for b in (4,8,16)]
    assert set(data["arms"])==set(ARMS)
    for kind,published in data["arms"].items():
        all_arms=[r["arms"][kind] for r in records]
        primary=[r["arms"][kind] for r in late]
        rebuilt=independent_aggregate(all_arms)
        primary_stats=independent_aggregate(primary)
        for key,v in rebuilt.items():
            assert published[key]==v,(kind,key)
        assert published["late_source"]==primary_stats
        for family in families:
            assert published["by_family"][family]==independent_aggregate([
                r["arms"][kind] for r in records if r["family"]==family])
        values=[primary_stats["certificates"],
                primary_stats["future_correct"],
                primary_stats["false_accept"],
                primary_stats["full_public_checks"],
                rebuilt["certificates"],
                rebuilt["future_correct"],
                rebuilt["false_accept"],
                rebuilt["full_public_checks"]]
        assert values==pinned["arm_outcomes"][kind],(kind,values)
        if not kind.startswith("greedy"):
            assert rebuilt["certificates"]==rebuilt["future_correct"]
            assert rebuilt["false_accept"]==0
    comparisons={
        "prefix4_vs_unscreened4":("unscreened_4","prefix_4","certified"),
        "prefix4_vs_prefix_original4":(
            "prefix_original_4","prefix_4","certified"),
        "prefix4_vs_prefix_hash4":("prefix_hash_4","prefix_4","certified"),
        "prefix8_vs_prefix4":("prefix_4","prefix_8","certified"),
        "prefix16_vs_prefix8":("prefix_8","prefix_16","certified"),
        "prefix16_vs_unscreened16":(
            "unscreened_16","prefix_16","certified"),
        "greedy4_vs_prefix4_future_correct":(
            "prefix_4","greedy_4","future_correct"),
    }
    assert set(data["primary_paired_tests"])==set(comparisons)
    for name,(a,b,measure) in comparisons.items():
        calc=paired_records(records,a,b,measure)
        original=data["primary_paired_tests"][name]
        pinned_test=pinned["paired_tests"][name]
        for field,value in calc.items():
            if field=="exact_unadjusted_p":
                assert abs(value-original[field])<1e-14
                assert abs(value-pinned_test["p"])<1e-14
            else:
                assert value==original[field]
        assert calc["gains"]==pinned_test["gains"]
        assert calc["losses"]==pinned_test["losses"]
    for family in families:
        for prefix in ("unscreened","prefix"):
            counts=[data["arms"][f"{prefix}_{k}"]["by_family"][family][
                "certificates"] for k in (4,8,16)]
            assert counts==sorted(counts)
    return {
        "full_test_cases":288,
        "primary_distinct_complete_public_sources":192,
        "case_arm_decisions_audited":len(records)*len(ARMS),
        "hypothetical_output_sha256_entries_audited":len(records)*16,
        "retrospective_16_check_source_verifications_charged":4608,
        "seven_paired_contrasts_recomputed":7,
        "prefix4_late_certificates":28,
        "prefix8_late_certificates":30,
        "prefix16_late_certificates":47,
        "certified_future_false_accepts":0,
        "preregistered_two_control_rank_specific_hypothesis_passed":False,
    }


def main():
    p=argparse.ArgumentParser()
    input_group=p.add_mutually_exclusive_group()
    input_group.add_argument("--full",type=Path)
    input_group.add_argument("--archive",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    pinned=pinned_receipt()
    raw=None
    if args.full:
        raw=args.full.read_bytes()
    if args.archive:
        with ZipFile(args.archive) as archive:
            assert archive.namelist()==["nor17_full.json"]
            assert archive.testzip() is None
            raw=archive.read("nor17_full.json")
    if raw is not None:
        assert len(raw)==pinned["original_full_json_bytes"]
        assert sha256(raw).hexdigest()==FULL_SHA
        result=full_audit(json.loads(raw),pinned)
        print("NOR17 INDEPENDENT SOURCE-LOCKED PROOF CERTIFICATE AUDIT PASSED",result)
    else:
        print("NOR17 FROZEN FIRST RUN RECEIPT VALID")
    if args.check:
        assert pinned["first_green_run"].startswith("https://github.com/")


if __name__=="__main__":
    main()
