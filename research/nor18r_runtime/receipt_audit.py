"""Independent NOR-18R source-only artifact and measured arithmetic auditor.

Uses no benchmark module or benchmark trained model. Reads archived original
public fixture and first-hosted timing JSON, verifies their immutable byte
SHA-256 hashes, reconstructs all public masks with the ORIGINAL native
corpus checker, replays candidate query outputs, inverse operand widths,
per-case sample medians, and all registered aggregate speed/memory claims.

Measured wall-clock times are observed data. This audit checks the samples
and arithmetic but cannot reproduce identical nanosecond timings on a
different CPU run. It does not consult a planted program or withheld future.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from hashlib import sha256
import argparse
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

RECEIPT=HERE/"results"/"nor18r_first_frozen_receipt.json"
ORIGINAL_RUN_SHA="59d48c8c9fe1bdf80ee2662d0c18f069f8414a92d48ad7980e3bb19b2a10ad98"
ORIGINAL_FIXTURE_SHA="5d75f2971d8b94daa9b65caae68e65921f2a90cd267481c7bb80a881e937d186"
ORIGINAL_NOR18_SHA="9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b"
MODES=("independent16","cached16","prefix160","prefix176")
SEED=202611181816


def ref_receipt():
    r=json.loads(RECEIPT.read_text())
    assert r["schema"]=="nor18r.first_hosted_source_locked_cpu_and_bigint_receipt.v1"
    assert r["original_benchmark_json_sha256"]==ORIGINAL_RUN_SHA
    assert r["original_benchmark_bytes"]==580813
    assert r["public_fixture_json_sha256"]==ORIGINAL_FIXTURE_SHA
    assert r["public_fixture_bytes"]==1306733
    assert r["source_nor18_original_json_sha256"]==ORIGINAL_NOR18_SHA
    assert r["source_cases"]==64 and r["source_cohorts"]==4
    assert r["primary_preregistered_passed"] is True
    assert r["primary_source_cases_cached_faster"]==64
    assert r["primary_median_paired_end_to_end_speedup_cached_vs_independent"]==4.328780845629943
    assert r["modes"]==list(MODES)
    return r


def fixture_public_case(item):
    assert set(item)=={
        "cohort","original_index","family","original_full_public_sha256",
        "original_model_input_sha256","public_header","demos",
        "public_query_input","original_exact_full_accept_mask",
        "original_public_future_output_identifiable",
    }
    assert item["cohort"] in (0,1,2,3)
    index=item["original_index"]
    assert index//72==item["cohort"]
    off=index%72
    assert (item["family"]=="late_source" and off in range(8)
            or item["family"]=="signed" and off in range(48,52)
            or item["family"]=="rational" and off in range(60,64))
    assert len(item["demos"])==2
    assert all(len(f)==len(y)==193 and f[0]==1
               for f,y in item["demos"])
    assert len(item["public_query_input"])==257
    assert item["public_query_input"][0]==1
    assert set(item["public_header"])=={"base","delay","spacing","amplitude"}
    h=item["public_header"]
    assert h["base"] in corpus.OPS
    assert h["delay"] in corpus.DELAYS
    assert h["spacing"] in corpus.SPACINGS
    assert h["amplitude"] in corpus.AMPLITUDES
    digest=sha256(json.dumps(
        [item["demos"],item["public_query_input"]],
        separators=(",",":")).encode()).hexdigest()
    assert digest==item["original_full_public_sha256"]
    assert len(item["original_model_input_sha256"])==64
    demos=tuple((tuple(f),tuple(y)) for f,y in item["demos"])
    c=corpus.Case(str(index),item["family"],demos,
                  tuple(item["public_query_input"]),(),None)
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    return c,header


def original_native_accept_mask(c,header):
    return sum(1<<j for j in range(16)
               if corpus.verify_template(
                   c,header,corpus.OPS[j//4],corpus.OPS[j%4]))


def original_native_future_predictions(c,header):
    return [corpus.execute(c.query,(header.base,header.delay,header.spacing,
                                    header.amplitude,corpus.OPS[j//4],
                                    corpus.OPS[j%4]))
            for j in range(16)]


def inverse_observed_work(f):
    assert f[0]==1
    r=[1]
    operations=weighted=zeros=max_observed_bits=0
    max_output_bits=1
    max_bytes=sys.getsizeof(1)
    for n in range(1,len(f)):
        value=0
        for k in range(1,n+1):
            a=f[k];b=r[n-k]
            operations+=1
            if a==0 or b==0:
                zeros+=1
            aa=max(1,abs(a).bit_length())
            bb=max(1,abs(b).bit_length())
            weighted+=aa*bb
            max_observed_bits=max(max_observed_bits,aa,bb)
            value+=a*b
        r.append(-value)
        max_output_bits=max(max_output_bits,abs(value).bit_length())
        max_bytes=max(max_bytes,sys.getsizeof(r[-1]))
    assert tuple(r)==corpus.d.operate(f,"R")
    return {
        "multiplications":operations,
        "operand_bitwidth_products":weighted,
        "zero_operand_multiplications":zeros,
        "max_operand_bits":max_observed_bits,
        "max_inverse_coefficient_bits":max_output_bits,
        "max_inverse_coefficient_python_int_bytes":max_bytes,
    }


def independent_bigint_check(c,h,record):
    per_degree={
        d:[inverse_observed_work(f[:d+1]) for f,y in c.demos]
        for d in (160,176,192)}
    q=inverse_observed_work(c.query)
    bigint=record["bigint_operand_width_ledger"]
    assert bigint["query_inverse_one_pass"]==q
    assert bigint["max_public_input_coefficient_bits"]==max(
        abs(v).bit_length() for f,y in c.demos for v in f)
    assert bigint["max_public_observed_output_coefficient_bits"]==max(
        abs(v).bit_length() for f,y in c.demos for v in y)
    assert bigint["max_query_source_coefficient_bits"]==max(
        abs(v).bit_length() for v in c.query)
    assert bigint["max_reciprocal_intermediate_bits"]==max(
        [q["max_inverse_coefficient_bits"]]+[
            x["max_inverse_coefficient_bits"] for d in per_degree for x in per_degree[d]])
    assert bigint["max_reciprocal_python_int_bytes"]==max(
        [q["max_inverse_coefficient_python_int_bytes"]]+[
            x["max_inverse_coefficient_python_int_bytes"]
            for d in per_degree for x in per_degree[d]])
    for mode in MODES:
        expected=bigint["per_mode"][mode]
        full_per_demo=(
            16*int(h.base=="R")+8
            if mode=="independent16" else
            1 if mode=="cached16" else
            sum(int(h.base=="R")+
                int(corpus.OPS[j//4]=="R")+
                int(corpus.OPS[j%4]=="R")
                for j in range(16)
                if record["prefix_public_survivor_masks"][
                    str(160 if mode=="prefix160" else 176)]&(1<<j))
        )
        prefix_degree=160 if mode=="prefix160" else (
            176 if mode=="prefix176" else None)
        assert expected["R_invocations_query"]==1
        assert expected["R_invocations_full_public_demos"]==2*full_per_demo
        assert expected["R_invocations_short_public_demos"]==(
            2 if prefix_degree else 0)
        operands=q["operand_bitwidth_products"]
        products=q["multiplications"]
        zeros=q["zero_operand_multiplications"]
        for x in per_degree[192]:
            operands+=full_per_demo*x["operand_bitwidth_products"]
            products+=full_per_demo*x["multiplications"]
            zeros+=full_per_demo*x["zero_operand_multiplications"]
        if prefix_degree:
            for x in per_degree[prefix_degree]:
                operands+=x["operand_bitwidth_products"]
                products+=x["multiplications"]
                zeros+=x["zero_operand_multiplications"]
        assert products==expected["R_integer_multiplications_executed"]
        assert operands==expected["R_operand_bitwidth_product_proxy"]
        assert zeros==expected["R_zero_operand_products_executed"]


def approximately(x,y):
    return math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9)


def independent_bootstrap(records,draws=4096):
    blocks=defaultdict(list)
    for r in records:
        blocks[r["cohort"]].append(r)
    assert set(blocks)=={0,1,2,3}
    assert all(len(v)==16 for v in blocks.values())
    randomizer=random.Random(SEED+17)
    sample_medians=[]
    for _ in range(draws):
        chosen=[r for i in (randomizer.randrange(4) for k in range(4))
                for r in blocks[i]]
        sample_medians.append(statistics.median(
            x["speedup_independent_over_cached_total"] for x in chosen))
    sample_medians.sort()
    return (sample_medians[int(.025*(draws-1))],
            sample_medians[int(.975*(draws-1))])


def audit(public_json,full_json,receipt):
    f=json.loads(public_json)
    z=json.loads(full_json)
    assert f["schema"]=="nor18r.public_fixture_64_cases_from_original_288.v1"
    assert f["NOR18_original_evidence_sha256"]==ORIGINAL_NOR18_SHA
    assert f["frozen_checkpoint_sha256"]==receipt["source_original_checkpoint_sha256"]
    assert f["public_only"] and f["no_hidden_query_outputs_or_planted_program_identities"]
    assert z["schema"]=="nor18r.executable_cpu_bigint_memory_source_locked.v1"
    assert z["fixture_cases"]==64 and len(z["per_case"])==64
    assert z["original_NOR18_full_JSON_SHA256"]==ORIGINAL_NOR18_SHA
    assert z["fixture_sha256"]==ORIGINAL_FIXTURE_SHA
    assert z["timed_repetitions"]==7
    assert z["untimed_warmups_per_case_per_algorithm"]==2
    assert z["modes"]==list(MODES)
    assert z["source_families"]=={"late_source":32,"signed":16,"rational":16}
    assert z["primary_cohort_block_bootstrap"]["draws"]==4096
    assert z["primary_preregistered_speedup_threshold"]==1.5
    assert z["primary_preregistered_faster_case_threshold"]==48
    assert z["primary_preregistered_passed"]
    assert len(f["cases"])==64
    assert len({item["original_full_public_sha256"] for item in f["cases"]})==64
    assert Counter(item["family"] for item in f["cases"])=={
        "late_source":32,"signed":16,"rational":16}
    counts=Counter()
    for source,obs in zip(f["cases"],z["per_case"],strict=True):
        c,header=fixture_public_case(source)
        counts[c.family]+=1
        assert obs["cohort"]==source["cohort"]
        assert obs["original_index"]==source["original_index"]
        assert obs["family"]==source["family"]
        assert obs["full_public_input_sha256"]==source["original_full_public_sha256"]
        assert obs["public_header"]==source["public_header"]
        assert obs["reference_full_public_accept_mask"]==source[
            "original_exact_full_accept_mask"]
        assert obs["reference_unanimous_future_output"]==source[
            "original_public_future_output_identifiable"]
        assert obs["algorithm_result_parity_all_modes"]
        mask=original_native_accept_mask(c,header)
        assert mask==obs["reference_full_public_accept_mask"]
        predictions=original_native_future_predictions(c,header)
        true_unanimity=len({predictions[j] for j in range(16)
                            if mask&(1<<j)})==1
        assert true_unanimity==obs["reference_unanimous_future_output"]
        for degree in (160,176):
            from_prefix=0
            for j in range(16):
                p=(header.base,header.delay,header.spacing,header.amplitude,
                   corpus.OPS[j//4],corpus.OPS[j%4])
                if all(corpus.execute(inp[:degree+1],p)==out[:degree+1]
                       for inp,out in c.demos):
                    from_prefix|=1<<j
            assert from_prefix==obs["prefix_public_survivor_masks"][str(degree)]
            assert mask&from_prefix==mask
        assert set(obs["measured_ns"])==set(MODES)
        assert set(obs["median_ns"])==set(MODES)
        assert set(obs["untimed_tracemalloc"])==set(MODES)
        for mode in MODES:
            records=obs["measured_ns"][mode]
            assert len(records)==7
            for x in records:
                assert set(x)=={"query_ns","verification_ns","certificate_ns","total_ns"}
                assert all(type(value) is int and value>=0 for value in x.values())
                assert x["total_ns"]==x["query_ns"]+x["verification_ns"]+x["certificate_ns"]
                assert x["total_ns"]>0
            for key in ("query_ns","verification_ns","certificate_ns","total_ns"):
                assert approximately(statistics.median(x[key] for x in records),
                                     obs["median_ns"][mode][key])
            mem=obs["untimed_tracemalloc"][mode]
            assert mem["mask"]==mask
            assert 0<=mem["python_tracemalloc_current_bytes"]<=mem[
                "python_tracemalloc_peak_bytes"]
        speed=obs["median_ns"]["independent16"]["total_ns"]/obs[
            "median_ns"]["cached16"]["total_ns"]
        assert approximately(speed,obs["speedup_independent_over_cached_total"])
        independent_bigint_check(c,header,obs)
    assert counts=={"late_source":32,"signed":16,"rational":16}
    ratios=[r["speedup_independent_over_cached_total"] for r in z["per_case"]]
    assert approximately(statistics.median(ratios),z[
        "primary_median_paired_total_speedup"])
    assert approximately(statistics.median(ratios),receipt[
        "primary_median_paired_end_to_end_speedup_cached_vs_independent"])
    assert sum(x>1 for x in ratios)==z["primary_cached_faster_cases"]==64
    lo,hi=independent_bootstrap(z["per_case"])
    ci=z["primary_cohort_block_bootstrap"]
    assert approximately(ci["percentile_2_5"],lo)
    assert approximately(ci["percentile_97_5"],hi)
    assert approximately(lo,receipt["cohort_block_bootstrap_95"][0])
    assert approximately(hi,receipt["cohort_block_bootstrap_95"][1])
    summary=z["summary_all"]
    for mode in MODES:
        median_total=statistics.median(r["median_ns"][mode]["total_ns"]
                                       for r in z["per_case"])/1e6
        median_verification=statistics.median(
            r["median_ns"][mode]["verification_ns"] for r in z["per_case"])/1e6
        median_peak=statistics.median(
            r["untimed_tracemalloc"][mode]["python_tracemalloc_peak_bytes"]
            for r in z["per_case"])/1024
        assert approximately(median_total,summary["per_mode_median_total_ms"][mode])
        assert approximately(median_total,receipt["per_mode_median_total_ms"][mode])
        assert approximately(median_verification,summary[
            "per_mode_median_verification_ms"][mode])
        assert approximately(median_peak,summary[
            "per_mode_median_temporary_tracemalloc_KiB"][mode])
        assert approximately(median_peak,receipt[
            "per_mode_median_python_tracemalloc_peak_KiB"][mode])
        for output_key,source_key,receipt_key in (
            ("per_mode_total_R_integer_multiplications",
             "R_integer_multiplications_executed",
             "per_mode_total_native_R_integer_products"),
            ("per_mode_total_R_operand_width_product_proxy",
             "R_operand_bitwidth_product_proxy",
             "per_mode_total_R_operand_bitwidth_product_proxy")):
            expected=sum(r["bigint_operand_width_ledger"]["per_mode"][mode][
                source_key] for r in z["per_case"])
            assert summary[output_key][mode]==expected==receipt[receipt_key][mode]
        rss=z["isolated_process_RSS_KiB"][mode]
        assert rss["mode"]==mode and rss["cases"]==64
        assert rss["process_exit_separate"]
        assert rss["baseline_peak_rss_KiB_before_algorithm"]==receipt[
            "memory_isolated_process_baseline_RSS_KiB"][mode]
        assert rss["process_peak_rss_KiB_after_algorithm"]==receipt[
            "memory_isolated_process_peak_RSS_KiB"][mode]
        assert rss["max_rss_increase_KiB"]==0

    for fam,source_summary in z["summary_by_family"].items():
        selected=[x for x in z["per_case"] if x["family"]==fam]
        assert source_summary["cases"]==len(selected)
        assert approximately(statistics.median(
            x["speedup_independent_over_cached_total"] for x in selected),
            source_summary["median_paired_total_speedup"])
        assert approximately(source_summary["median_paired_total_speedup"],
                             receipt["family_median_paired_speedup"][fam])

    for flag in (
        "all_source_inputs_from_original_NOR18_288_case_test",
        "exact_public_mask_and_certificate_match_original_for_every_mode",
        "future_outputs_and_private_generator_programs_not_in_fixture",
        "native_independent_full_verifier_is_unmodified",
        "cached_public_verifier_uses_existing_NOR17_compound_series",
        "fixed_public_query_recomputed_for_each_mode",
        "time_includes_python_bigint_implementation_effects",
        "input_serialization_and_cohort_generation_excluded_from_timing",
        "cohort_bootstrap_not_independent_research_replication",
        "reciprocal_integer_operand_width_products_are_cost_proxy",
        "RSS_and_tracemalloc_measured_in_separate_untimed_passes",
        "no_general_theorem_proving_or_GPU_speedup_claim",
    ):
        assert z["scope"][flag],flag
    return {
        "public_only_source_cases":64,
        "distinct_source_hashes_verified":64,
        "source_cohorts":4,
        "exact_native_masks_and_certificates_recomputed":64,
        "timed_samples_checked":64*4*7,
        "untimed_memory_records_checked":64*4,
        "bigint_operand_width_ledgers_replayed":64,
        "four_cohort_bootstrap_recomputed":4096,
        "registered_performance_criterion_passed":True,
        "paired_complete_CPU_speedup":statistics.median(ratios),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--full",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    pinned=ref_receipt()
    if args.archive:
        assert args.fixture is None and args.full is None
        with ZipFile(args.archive) as z:
            assert sorted(z.namelist())==[
                "nor18r_full.json","nor18r_public_fixture.json"]
            assert z.testzip() is None
            original_fixture=z.read("nor18r_public_fixture.json")
            original_run=z.read("nor18r_full.json")
    elif args.fixture and args.full:
        original_fixture=args.fixture.read_bytes()
        original_run=args.full.read_bytes()
    else:
        print("NOR18R FIRST ORIGINAL PUBLIC FIXTURE / BENCHMARK HASH RECEIPT VALID")
        return
    assert len(original_fixture)==pinned["public_fixture_bytes"]
    assert len(original_run)==pinned["original_benchmark_bytes"]
    assert sha256(original_fixture).hexdigest()==ORIGINAL_FIXTURE_SHA
    assert sha256(original_run).hexdigest()==ORIGINAL_RUN_SHA
    report=audit(original_fixture,original_run,pinned)
    print("NOR18R INDEPENDENT CPU BIGINT MEMORY AUDIT PASSED",report)
    if args.check:
        assert report["source_cohorts"]==4
        assert report["timed_samples_checked"]==1792


if __name__=="__main__":
    main()
