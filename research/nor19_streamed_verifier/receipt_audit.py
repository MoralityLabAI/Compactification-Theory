"""NOR-19 independent original source, timing and memory receipt audit.

Does not import the NOR-19 benchmark or any new streaming implementation.
Reconstructs all exact public accepted masks and unique hypothetical future
certificates using the original RRC25 public checker, checks all first-
hosted nanosecond samples and untimed allocation profiles, independently
replays bigint source coefficient operand widths, and regenerates the
four-cohort bootstrap over original source records.

Measured times and traced allocations remain observations; this audit
cannot prove that another CPU will reproduce identical values.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor11_attractor_controller import controller as nor11

RECEIPT=HERE/"results"/"nor19_first_frozen_receipt.json"
FULL_SHA="16d6ce9acb042aff76d48f46b7d07ff76a4efca1ade26314cdc52926eb0e1a16"
FIXTURE_SHA="97fc9ccd3830bc666e6af6d0a86d3b000ef2b8d0a7a29a406ff38fbd08707f3f"
OLD_SHA="9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b"
MODEL_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
SEED=202611191900
MODES=("independent16","cached16","stream16_queryfirst",
       "stream16_verifyfirst","chunk4_verifyfirst")
PRIMARY="stream16_verifyfirst"


def eq_num(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-8)


def receipt():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor19.frozen_first_hosted_streamed_memory_cpu_receipt.v1"
    assert r["original_fixture_sha256"]==FIXTURE_SHA
    assert r["original_full_result_sha256"]==FULL_SHA
    assert r["original_fixture_bytes"]==1290607
    assert r["original_full_result_bytes"]==680861
    assert r["source_original_NOR18_sha256"]==OLD_SHA
    assert r["source_RRC25_checkpoint_sha256"]==MODEL_SHA
    assert r["source_cases"]==64 and r["source_cohorts"]==4
    assert r["timed_repetitions_per_mode"]==7
    assert r["warmups_per_mode"]==2
    assert r["algorithms"]==list(MODES)
    assert r["primary"]==PRIMARY
    assert r["primary_registered_pass"]
    assert r["primary_lower_peak_cases"]==64
    assert r["bootstrap_draws"]==4096
    return r


def source_from_public_only(row):
    allowed={"cohort","original_index","family","original_full_public_sha256",
             "original_model_input_sha256","public_header","demos",
             "public_query_input","original_exact_full_accept_mask",
             "original_public_future_output_identifiable"}
    assert set(row)==allowed
    rep=row["cohort"]
    assert rep in (0,1,2,3)
    ix=row["original_index"]
    assert ix//72==rep
    pos=ix%72
    assert ((row["family"]=="late_source" and 8<=pos<16)
         or (row["family"]=="signed" and 52<=pos<56)
         or (row["family"]=="rational" and 64<=pos<68))
    h=row["public_header"]
    assert set(h)=={"base","delay","spacing","amplitude"}
    assert h["base"] in corpus.OPS
    assert h["delay"] in corpus.DELAYS and h["spacing"] in corpus.SPACINGS
    assert h["amplitude"] in corpus.AMPLITUDES
    demos=tuple((tuple(f),tuple(y)) for f,y in row["demos"])
    assert len(demos)==2 and all(len(f)==len(y)==193 and f[0]==1 for f,y in demos)
    query=tuple(row["public_query_input"])
    assert len(query)==257 and query[0]==1
    c=corpus.Case(f"nor19-independent-{ix}",row["family"],
                  demos,query,(),None)
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    public_digest=sha256(json.dumps([c.visible,c.query],
                          separators=(",",":")).encode()).hexdigest()
    assert public_digest==row["original_full_public_sha256"]
    assert nor11.public_digest(c,header)==row["original_model_input_sha256"]
    assert c.future==() and c.program is None
    return c,header


def public_native_mask_and_output(c,h):
    accepted=0
    futures=[]
    for j in range(16):
        program=(h.base,h.delay,h.spacing,h.amplitude,
                 corpus.OPS[j//4],corpus.OPS[j%4])
        future=corpus.execute(c.query,program)
        futures.append(future)
        if corpus.verify_template(c,h,corpus.OPS[j//4],corpus.OPS[j%4]):
            accepted|=1<<j
    assert accepted
    return accepted,len({futures[j] for j in range(16)
                          if accepted&(1<<j)})==1


def one_inverse(f):
    """Repeat original Python recurrence without importing benchmark."""
    assert f[0]==1
    inv=[1]
    total_multiplications=0
    bit_width_products=0
    for degree in range(1,len(f)):
        value=0
        for k in range(1,degree+1):
            a=f[k]
            b=inv[degree-k]
            total_multiplications+=1
            bit_width_products+=max(1,abs(a).bit_length())*max(
                1,abs(b).bit_length())
            value+=a*b
        inv.append(-value)
    assert tuple(inv)==corpus.d.operate(f,"R")
    return total_multiplications,bit_width_products


def original_bigint_work(c,h,mode):
    inputs=[f for f,y in c.demos]
    costs=[one_inverse(f) for f in inputs]
    q=one_inverse(c.query)
    repetitions=(8+16*int(h.base=="R")) if mode=="independent16" else 1
    assert repetitions>=1
    return {
        "R_full_public_demo_invocations":2*repetitions,
        "R_public_query_invocations":1,
        "R_integer_products":q[0]+repetitions*sum(z[0] for z in costs),
        "R_operand_bitwidth_product_proxy":
            q[1]+repetitions*sum(z[1] for z in costs),
    }


def bootstrap(source,key,draws=4096):
    blocks=defaultdict(list)
    for entry in source:
        blocks[entry["cohort"]].append(entry)
    assert set(blocks)=={0,1,2,3}
    assert all(len(v)==16 for v in blocks.values())
    rng=random.Random(SEED+13)
    estimates=[]
    for _ in range(draws):
        selected=[z for i in (rng.randrange(4) for k in range(4))
                  for z in blocks[i]]
        estimates.append(statistics.median(x[key] for x in selected))
    estimates.sort()
    return {
        "draws":draws,
        "unit":"NOR18 original source cohort of 16 previously unbenchmarked cases",
        "percentile_2_5":estimates[int(.025*(draws-1))],
        "percentile_97_5":estimates[int(.975*(draws-1))],
    }


def median_from(source,fn):
    return statistics.median(fn(x) for x in source)


def audit(fixture_raw,result_raw,frozen):
    fixture=json.loads(fixture_raw)
    result=json.loads(result_raw)
    assert fixture["schema"]=="nor19.previously_untimed_public_source_64.v1"
    assert fixture["source_original_NOR18_sha256"]==OLD_SHA
    assert fixture["original_RRC25_checkpoint_sha256"]==MODEL_SHA
    assert fixture["previously_timed_NOR18R_source_indices_excluded"]
    assert fixture["public_only"]
    assert fixture["private_program_identities_and_query_outputs_absent"]
    assert result["schema"]=="nor19.streaming_memory_bounded_public_verifier.v1"
    assert result["original_NOR18_full_SHA256"]==OLD_SHA
    assert result["fixture_schema"]==fixture["schema"]
    assert result["fixture_sha256"]==FIXTURE_SHA
    assert result["case_count"]==len(result["records"])==64
    assert result["cohorts"]==4
    assert result["source_families"]=={"late_source":32,"signed":16,"rational":16}
    assert result["measured_repetitions"]==7 and result["warmups"]==2
    assert result["algorithms"]==list(MODES)
    assert result["primary"]==PRIMARY
    assert result["primary_median_peak_ratio_threshold"]==.8
    assert result["primary_faster_than_native_threshold"]==2.
    assert result["primary_memory_win_case_threshold"]==48
    assert result["primary_passed"]
    assert set(result["materialized_demo_candidate_buffers_at_once_design"])==set(MODES)
    assert result["materialized_demo_candidate_buffers_at_once_design"]["cached16"]==32
    assert result["materialized_demo_candidate_buffers_at_once_design"]["stream16_verifyfirst"]==0
    assert result["materialized_demo_candidate_buffers_at_once_design"]["chunk4_verifyfirst"]==4
    assert len(fixture["cases"])==64
    assert len({r["original_full_public_sha256"] for r in fixture["cases"]})==64
    assert Counter(r["family"] for r in fixture["cases"])=={
        "late_source":32,"signed":16,"rational":16}
    assert len({z["public_sha256"] for z in result["records"]})==64
    for flag in (
        "complete_public_original_masks_and_certificates_unchanged",
        "previous_NOR18R_measured_cases_excluded",
        "all_fixture_sources_match_original_NOR18_byte_locked_archive",
        "all_five_query_simulations_recomputed_independently",
        "withheld_true_future_and_program_identity_absent_from_fixture",
        "full_measured_times_include_query_and_verification",
        "tracemalloc_is_separate_untimed_execution",
        "memory_buffer_counts_are_design_bounds_not_bytes",
        "block_bootstrap_is_conditional_on_four_existing_source_cohorts",
        "hosted_CPU_is_not_cross_platform_guarantee",
    ):
        assert result["scope"][flag],flag
    for item,record in zip(fixture["cases"],result["records"],strict=True):
        c,h=source_from_public_only(item)
        assert item["original_full_public_sha256"]==record["public_sha256"]
        assert record["source_header"]==item["public_header"]
        assert record["cohort"]==item["cohort"]
        assert record["family"]==item["family"]
        assert record["original_index"]==item["original_index"]
        native_mask,unanimity=public_native_mask_and_output(c,h)
        assert native_mask==item["original_exact_full_accept_mask"]
        assert unanimity==item["original_public_future_output_identifiable"]
        assert record["reference_mask"]==native_mask
        assert record["reference_unanimous"]==unanimity
        assert record["exact_mask_and_certificate_parity"]
        assert set(record["mode_samples_ns"])==set(MODES)
        assert set(record["mode_medians_ns"])==set(MODES)
        assert set(record["complete_tracemalloc"])==set(MODES)
        assert set(record["verifier_only_tracemalloc"])==set(MODES)
        assert set(record["reciprocal_bigint_work"])==set(MODES)
        for mode in MODES:
            assert record["reciprocal_bigint_work"][mode]==original_bigint_work(c,h,mode)
            samples=record["mode_samples_ns"][mode]
            assert len(samples)==7
            for sample in samples:
                assert set(sample)=={
                    "query_ns","verification_ns","certificate_ns","total_ns"}
                assert all(type(v) is int and v>=0 for v in sample.values())
                assert sample["total_ns"]==(
                    sample["query_ns"]+sample["verification_ns"]+
                    sample["certificate_ns"])
                assert sample["query_ns"]>0 and sample["verification_ns"]>0
            for field in ("query_ns","verification_ns","certificate_ns","total_ns"):
                expected=median_from(samples,lambda x:x[field])
                assert eq_num(record["mode_medians_ns"][mode][field],expected)
            mem=record["complete_tracemalloc"][mode]
            assert mem["mask"]==native_mask
            assert (0<=mem["python_current_allocation_bytes"]<=
                    mem["python_peak_allocation_bytes"])
            assert mem["python_peak_allocation_bytes"]>0
            only=record["verifier_only_tracemalloc"][mode]
            assert only["mask"]==native_mask
            assert (0<=only["python_verifier_only_current_bytes"]<=
                    only["python_verifier_only_peak_bytes"])
        ratio=(record["complete_tracemalloc"][PRIMARY][
            "python_peak_allocation_bytes"]/
            record["complete_tracemalloc"]["cached16"][
                "python_peak_allocation_bytes"])
        speed=(record["mode_medians_ns"]["independent16"]["total_ns"]/
               record["mode_medians_ns"][PRIMARY]["total_ns"])
        assert eq_num(ratio,record["primary_peak_ratio_vs_cached"])
        assert eq_num(speed,record["primary_native_speedup"])
        assert record["streaming_lower_peak_than_cached"]==(ratio<1)
    rows=result["records"]
    ratios=[r["primary_peak_ratio_vs_cached"] for r in rows]
    speeds=[r["primary_native_speedup"] for r in rows]
    median_ratio=statistics.median(ratios)
    median_speed=statistics.median(speeds)
    wins=sum(x<1 for x in ratios)
    assert eq_num(median_ratio,result[
        "primary_median_complete_peak_ratio_vs_cached"])
    assert eq_num(median_ratio,frozen[
        "primary_median_complete_python_peak_ratio_stream_vs_cache"])
    assert eq_num(median_speed,result[
        "primary_median_total_native_over_stream_speedup"])
    assert eq_num(median_speed,frozen[
        "primary_median_native_over_streamed_cpu_speedup"])
    assert wins==result["primary_lower_peak_cases"]==frozen["primary_lower_peak_cases"]
    assert (median_ratio<=.8 and median_speed>=2. and wins>=48)==result[
        "primary_passed"]
    for field,receipt_key in (
        ("primary_peak_ratio_vs_cached","block_bootstrap_95_stream_over_cached_python_peak_ratio"),
        ("primary_native_speedup","block_bootstrap_95_native_over_streamed_cpu_speedup"),
    ):
        recomputed=bootstrap(rows,field)
        original=result[
            "primary_memory_block_bootstrap" if field.startswith("primary_peak")
            else "primary_speed_block_bootstrap"]
        assert recomputed["draws"]==original["draws"]==4096
        for quant in ("percentile_2_5","percentile_97_5"):
            assert eq_num(recomputed[quant],original[quant])
        assert eq_num(recomputed["percentile_2_5"],frozen[receipt_key][0])
        assert eq_num(recomputed["percentile_97_5"],frozen[receipt_key][1])
    summaries={"all":rows}
    summaries.update({fam:[x for x in rows if x["family"]==fam]
                      for fam in ("late_source","signed","rational")})
    for name,group in summaries.items():
        summary=(result["summary_all"] if name=="all" else
                 result["summary_by_family"][name])
        assert summary["cases"]==len(group)
        assert eq_num(summary["median_streamed_peak_vs_cached_ratio"],
                      median_from(group,lambda x:x["primary_peak_ratio_vs_cached"]))
        assert eq_num(summary["median_native_over_primary_streamed_total_speedup"],
                      median_from(group,lambda x:x["primary_native_speedup"]))
        assert summary["streamed_lower_peak_cases"]==sum(
            x["primary_peak_ratio_vs_cached"]<1 for x in group)
        for mode in MODES:
            for key,src,scale in (
                ("median_complete_ms","mode_medians_ns",1e6),
                ("median_verifier_ms","mode_medians_ns",1e6),
                ("median_complete_peak_KiB","complete_tracemalloc",1024),
                ("median_verifier_only_peak_KiB","verifier_only_tracemalloc",1024),
            ):
                if key=="median_complete_ms":
                    val=median_from(group,lambda x:x[src][mode]["total_ns"])/scale
                elif key=="median_verifier_ms":
                    val=median_from(group,lambda x:x[src][mode]["verification_ns"])/scale
                elif key=="median_complete_peak_KiB":
                    val=median_from(group,lambda x:x[src][mode][
                        "python_peak_allocation_bytes"])/scale
                else:
                    val=median_from(group,lambda x:x[src][mode][
                        "python_verifier_only_peak_bytes"])/scale
                assert eq_num(summary[key][mode],val)
                if name=="all":
                    receipt_key={
                        "median_complete_ms":"per_mode_median_complete_ms",
                        "median_verifier_ms":"per_mode_median_public_verification_ms",
                        "median_complete_peak_KiB":"per_mode_median_complete_tracemalloc_KiB",
                        "median_verifier_only_peak_KiB":"per_mode_median_verifier_only_tracemalloc_KiB"}[key]
                    assert eq_num(val,frozen[receipt_key][mode])
            assert summary["reciprocal_bigint_products"][mode]==sum(
                entry["reciprocal_bigint_work"][mode]["R_integer_products"]
                for entry in group)
    for mode in MODES:
        rss=result["isolated_process_RSS_KiB"][mode]
        assert rss["mode"]==mode and rss["cases"]==64
        assert rss["separate_process"]
        assert rss["peak_RSS_after_all_cases_KiB"]==frozen[
            "per_mode_separate_process_peak_RSS_KiB"][mode]
        assert rss["baseline_peak_RSS_KiB"]==frozen[
            "per_mode_separate_process_baseline_RSS_KiB"][mode]
        assert rss["additional_max_RSS_KiB"]>=0
    return {
        "public_only_unused_source_cases_checked":64,
        "exact_native_masks_and_output_unanimity_recomputed":64,
        "first_hosted_timing_samples_recomputed":64*5*7,
        "separate_memory_profiles_checked":64*5*2,
        "reciprocal_bigint_work_profiles_replayed":64*5,
        "bootstrap_resamples_recomputed":2*4096,
        "primary_complete_streaming_peak_ratio":median_ratio,
        "primary_native_to_streaming_speedup":median_speed,
        "first_preregistered_primary_success":True,
    }


def main():
    p=argparse.ArgumentParser()
    input_group=p.add_mutually_exclusive_group()
    input_group.add_argument("--archive",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--full",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    frozen=receipt()
    public_bytes=result_bytes=None
    if args.archive:
        with ZipFile(args.archive) as z:
            assert sorted(z.namelist())==[
                "nor19_full.json","nor19_public_fixture.json"]
            assert z.testzip() is None
            public_bytes=z.read("nor19_public_fixture.json")
            result_bytes=z.read("nor19_full.json")
    elif args.fixture and args.full:
        public_bytes=args.fixture.read_bytes()
        result_bytes=args.full.read_bytes()
    elif args.fixture or args.full:
        p.error("requires both --fixture and --full")
    else:
        print("NOR19 ORIGINAL HOSTED FIRST-RESULT SOURCE RECEIPT VALID")
        return
    assert len(public_bytes)==frozen["original_fixture_bytes"]
    assert len(result_bytes)==frozen["original_full_result_bytes"]
    assert sha256(public_bytes).hexdigest()==FIXTURE_SHA
    assert sha256(result_bytes).hexdigest()==FULL_SHA
    result=audit(public_bytes,result_bytes,frozen)
    print("NOR19 INDEPENDENT EXACT PUBLIC MASK / CPU / MEMORY AUDIT PASSED",result)
    if args.check:
        assert result["first_hosted_timing_samples_recomputed"]==2240
        assert result["first_preregistered_primary_success"]


if __name__=="__main__":
    main()
