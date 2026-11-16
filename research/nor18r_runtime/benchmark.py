"""NOR-18R: executable source-locked CPU, Python memory and bigint audit.

Run under CPython 3.12 on a dedicated hosted Actions CPU runner. Four
public algorithms must return the same full sixteen-class accepted mask
and output-unanimity certificate on 64 public heldout NOR18 sources.

Measured timings include fresh public degree256 query output simulation
for every algorithm, separately reported from degree192 public
verification. No private query future output or planted source program
is present in the generated fixture or read by any benchmark method.

The prior NOR18 full JSON is used exclusively to verify original source
hashes and reference public mask / identifiability; it is never read by a
policy while executing. Bigint bit widths are observed from exact inverse
recurrences outside the timed phases. Memory allocations are measured in
separate untimed passes.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor11_attractor_controller import controller as nor11
from research.nor12_task_readout import readout as nor12
from research.nor16_public_query_consensus import experiment as nor16
from research.nor17_prefix_certificates import experiment as nor17
from research.nor18_costed_prefix import experiment as nor18

SCHEMA="nor18r.executable_cpu_bigint_memory_source_locked.v1"
FIXTURE_SCHEMA="nor18r.public_fixture_64_cases_from_original_288.v1"
NOR18_ZIP=ROOT/"research/nor18_costed_prefix/results/nor18_full.zip"
NOR18_SHA="9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b"
CHECKPOINT_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"
MODES=("independent16","cached16","prefix160","prefix176")
SAMPLES_PER_REP={"late_source":8,"signed":4,"rational":4}
COHORTS=4
SOURCES=64
TIMED_REPS=7
WARMUPS=2
BOOTSTRAPS=4096
SEED=202611181816
QUERY_DEGREE=256
PUBLIC_DEGREE=192


def serialized(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"))


def read_nor18():
    with ZipFile(NOR18_ZIP) as z:
        assert z.namelist()==["nor18_full.json"]
        assert z.testzip() is None
        source=z.read("nor18_full.json")
    assert len(source)==12896423
    assert sha256(source).hexdigest()==NOR18_SHA
    d=json.loads(source)
    assert d["source_checkpoint_sha256"]==CHECKPOINT_SHA
    assert d["all_cases"]==288
    assert len(d["records"])==288
    assert d["primary_late_distinct_complete_public_inputs"]==192
    return d


def fixture_from_locked_source():
    d=read_nor18()
    original=nor11.sample_cohorts(nor12.SPLIT_COUNTS)
    batches=nor18.fresh_cohorts(original)
    assert len(batches)==4 and len(d["cohorts"])==4
    selected=[]
    for rep,batch in enumerate(batches):
        assert len(batch)==72
        manifest=sha256("\n".join(
            nor16.full_public_digest(case) for case,h,public_digest in batch
        ).encode()).hexdigest()
        assert manifest==d["cohorts"][rep]["public_manifest_sha256"]
        for ix,(case,header,model_hash) in enumerate(batch):
            published=d["records"][rep*72+ix]
            assert case.family==published["family"]
            assert nor16.full_public_digest(case)==published["full_public_input_sha256"]
            assert nor11.public_digest(case,header)==model_hash==published["model_visible_sha256"]
        # Original cohort order: 48 late_source, 12 signed, 12 rational.
        chosen_indexes=list(range(8))+list(range(48,52))+list(range(60,64))
        assert Counter(batch[i][0].family for i in chosen_indexes)==SAMPLES_PER_REP
        for ix in chosen_indexes:
            case,head,model_hash=batch[ix]
            published=d["records"][rep*72+ix]
            demos=[[list(source[:PUBLIC_DEGREE+1]),
                    list(output[:PUBLIC_DEGREE+1])]
                   for source,output in case.demos]
            selected.append({
                "cohort":rep,
                "original_index":rep*72+ix,
                "family":case.family,
                "original_full_public_sha256":published["full_public_input_sha256"],
                "original_model_input_sha256":model_hash,
                "public_header":{"base":head.base,
                                 "delay":head.delay,
                                 "spacing":head.spacing,
                                 "amplitude":head.amplitude},
                "demos":demos,
                "public_query_input":list(case.query),
                "original_exact_full_accept_mask":published["retro_full_accepted_mask"],
                "original_public_future_output_identifiable":
                    published["certificate_possible_from_complete_public_mask"],
            })
    assert len(selected)==SOURCES
    assert len({s["original_full_public_sha256"] for s in selected})==64
    assert Counter(s["family"] for s in selected)=={
        "late_source":32,"signed":16,"rational":16}
    return {
        "schema":FIXTURE_SCHEMA,
        "NOR18_original_evidence_sha256":NOR18_SHA,
        "frozen_checkpoint_sha256":CHECKPOINT_SHA,
        "source_selection":"four cohorts, 8 late + 4 signed + 4 rational each",
        "cases":selected,
        "public_only":True,
        "no_hidden_query_outputs_or_planted_program_identities":True,
    }


def load_fixture(path):
    raw=Path(path).read_bytes()
    obj=json.loads(raw)
    assert obj["schema"]==FIXTURE_SCHEMA
    assert obj["NOR18_original_evidence_sha256"]==NOR18_SHA
    assert obj["frozen_checkpoint_sha256"]==CHECKPOINT_SHA
    assert obj["public_only"]
    assert obj["no_hidden_query_outputs_or_planted_program_identities"]
    assert len(obj["cases"])==SOURCES
    assert Counter(z["family"] for z in obj["cases"])=={
        "late_source":32,"signed":16,"rational":16}
    assert len({z["original_full_public_sha256"] for z in obj["cases"]})==64
    return obj


def unpack_case(row):
    h=row["public_header"]
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    demos=tuple((tuple(x),tuple(y)) for x,y in row["demos"])
    query=tuple(row["public_query_input"])
    assert len(demos)==2 and all(len(x)==len(y)==193 for x,y in demos)
    assert len(query)==257
    # Private source program and future query truth deliberately absent.
    c=corpus.Case(f"nor18r-{row['original_index']}",row["family"],
                  demos,query,(),None)
    assert nor16.full_public_digest(c)==row["original_full_public_sha256"]
    assert nor11.public_digest(c,header)==row["original_model_input_sha256"]
    return c,header


def mask_independent(c,h):
    """Unmodified native exact full verification of 16 templates."""
    return sum(1<<j for j in range(16)
               if corpus.verify_template(c,h,corpus.OPS[j//4],corpus.OPS[j%4]))


def mask_cached(c,h):
    """Two public 193-row batches, four shared unary transforms per demo."""
    predicted=[
        nor17.candidate_outputs(f,h,PUBLIC_DEGREE)
        for f,y in c.demos
    ]
    return sum(1<<j for j in range(16) if all(
        predicted[k][j]==c.demos[k][1] for k in range(2)))


def mask_prefix(c,h,degree):
    candidate_mask=nor17.prefix_mask(c,h,degree)
    return sum(1<<j for j in range(16)
               if candidate_mask&(1<<j)
               and corpus.verify_template(c,h,corpus.OPS[j//4],corpus.OPS[j%4]))


def evaluated_full_mask(c,h,mode):
    if mode=="independent16":
        return mask_independent(c,h)
    if mode=="cached16":
        return mask_cached(c,h)
    if mode=="prefix160":
        return mask_prefix(c,h,160)
    if mode=="prefix176":
        return mask_prefix(c,h,176)
    raise ValueError("unknown measured algorithm")


def outcome(c,h,mode,measure=False):
    """Compute query predictions and public mask. Never inspect c.future."""
    if measure:
        t0=time.perf_counter_ns()
    output=nor17.candidate_outputs(c.query,h,QUERY_DEGREE)
    if measure:
        t1=time.perf_counter_ns()
    mask=evaluated_full_mask(c,h,mode)
    if measure:
        t2=time.perf_counter_ns()
    assert 0<mask<65536
    y={output[j] for j in range(16) if mask&(1<<j)}
    unanimous=len(y)==1
    chosen=min(j for j in range(16) if mask&(1<<j)) if unanimous else None
    if measure:
        t3=time.perf_counter_ns()
        times={
            "query_ns":t1-t0,
            "verification_ns":t2-t1,
            "certificate_ns":t3-t2,
            "total_ns":t3-t0,
        }
        assert all(t>=0 for t in times.values())
        return {"mask":mask,"unanimous":unanimous,
                "chosen_class":chosen,"times":times}
    return {"mask":mask,"unanimous":unanimous,"chosen_class":chosen}


def exact_inverse_operand_ledger(f):
    """Replay the actual degree-n R formal-series recurrence, counting
    executed Python integer products and their observed operand bit widths.

    This is an operand-size PROXY. It measures real integer bit_length()
    values but does not estimate native GMP cost or CPython clock ticks.
    """
    assert f[0]==1 and len(f)>1
    inverse=[1]
    products=0
    width_products=0
    peak_operand_bits=0
    peak_output_bits=1
    max_int_bytes=sys.getsizeof(1)
    zero_operand_products=0
    for n in range(1,len(f)):
        value=0
        for k in range(1,n+1):
            a=f[k]
            b=inverse[n-k]
            products+=1
            if a==0 or b==0:
                zero_operand_products+=1
            abit=max(1,abs(a).bit_length())
            bbit=max(1,abs(b).bit_length())
            width_products+=abit*bbit
            peak_operand_bits=max(peak_operand_bits,abit,bbit)
            value+=a*b
        inverse.append(-value)
        peak_output_bits=max(peak_output_bits,abs(value).bit_length())
        max_int_bytes=max(max_int_bytes,sys.getsizeof(inverse[-1]))
    exact=corpus.d.operate(f,"R")
    assert tuple(inverse)==exact
    assert products==(len(f)-1)*len(f)//2
    return {
        "multiplications":products,
        "operand_bitwidth_products":width_products,
        "zero_operand_multiplications":zero_operand_products,
        "max_operand_bits":peak_operand_bits,
        "max_inverse_coefficient_bits":peak_output_bits,
        "max_inverse_coefficient_python_int_bytes":max_int_bytes,
    }


def reciprocal_invocation_count(header,mode,source_masks):
    """Number of R transforms in the actual public-demo verification."""
    per_check=lambda j:(
        int(header.base=="R")+
        int(corpus.OPS[j//4]=="R")+
        int(corpus.OPS[j%4]=="R")
    )
    if mode=="independent16":
        return 16*int(header.base=="R")+8
    if mode=="cached16":
        return 1
    degree=160 if mode=="prefix160" else 176
    prefix=source_masks[str(degree)]
    return 1+sum(per_check(j) for j in range(16) if prefix&(1<<j))


def integer_cost_ledger(c,h,source_masks):
    demo_inverse={
        str(deg):[exact_inverse_operand_ledger(f[:deg+1]) for f,y in c.demos]
        for deg in (160,176,192)
    }
    query_inverse=exact_inverse_operand_ledger(c.query)
    per_mode={}
    for mode in MODES:
        verification_full_invocations=(
            16*int(h.base=="R")+8 if mode=="independent16"
            else 1 if mode=="cached16" else sum(
                (int(h.base=="R")+
                 int(corpus.OPS[j//4]=="R")+
                 int(corpus.OPS[j%4]=="R"))
                for j in range(16)
                if source_masks[str(160 if mode=="prefix160" else 176)]&(1<<j))
        )
        prefix_degree=(160 if mode=="prefix160" else
                       176 if mode=="prefix176" else None)
        # Independent native checks reapply R per-candidate; cached full
        # and prefix batch apply it once per public demonstration.
        inv_counts_by_stage={
            "query":1,
            "prefix":1 if prefix_degree is not None else 0,
            "full":verification_full_invocations,
        }
        products=query_inverse["multiplications"]
        widths=query_inverse["operand_bitwidth_products"]
        zeros=query_inverse["zero_operand_multiplications"]
        for deg,mult in ((192,verification_full_invocations),
                         (prefix_degree,1 if prefix_degree else 0)):
            if not mult:continue
            for inv in demo_inverse[str(deg)]:
                products+=mult*inv["multiplications"]
                widths+=mult*inv["operand_bitwidth_products"]
                zeros+=mult*inv["zero_operand_multiplications"]
        per_mode[mode]={
            "R_invocations_query":1,
            "R_invocations_full_public_demos":2*verification_full_invocations,
            "R_invocations_short_public_demos":(2 if prefix_degree else 0),
            "R_integer_multiplications_executed":products,
            "R_operand_bitwidth_product_proxy":widths,
            "R_zero_operand_products_executed":zeros,
        }
    return {
        "per_mode":per_mode,
        "max_public_input_coefficient_bits":max(
            abs(t).bit_length() for f,y in c.demos for t in f),
        "max_public_observed_output_coefficient_bits":max(
            abs(t).bit_length() for f,y in c.demos for t in y),
        "max_query_source_coefficient_bits":max(
            abs(t).bit_length() for t in c.query),
        "max_reciprocal_intermediate_bits":max(
            [query_inverse["max_inverse_coefficient_bits"]]+
            [z["max_inverse_coefficient_bits"] for group in demo_inverse.values()
             for z in group]),
        "max_reciprocal_python_int_bytes":max(
            [query_inverse["max_inverse_coefficient_python_int_bytes"]]+
            [z["max_inverse_coefficient_python_int_bytes"]
             for group in demo_inverse.values() for z in group]),
        "query_inverse_one_pass":query_inverse,
    }


def ephemeral_memory_profile(c,h,mode):
    gc.collect()
    tracemalloc.start()
    try:
        result=outcome(c,h,mode)
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "python_tracemalloc_peak_bytes":peak,
        "python_tracemalloc_current_bytes":current,
        "mask":result["mask"],
    }


def median_ns(samples,name):
    return float(statistics.median(z[name] for z in samples))


def block_bootstrap(case_rows,draws=BOOTSTRAPS):
    blocks=defaultdict(list)
    for case in case_rows:
        blocks[case["cohort"]].append(case)
    assert set(blocks)=={0,1,2,3}
    assert all(len(x)==16 for x in blocks.values())
    rng=random.Random(SEED+17)
    values=[]
    for _ in range(draws):
        sample=[item for i in (rng.randrange(4) for j in range(4))
                for item in blocks[i]]
        ratios=[v["speedup_independent_over_cached_total"] for v in sample]
        values.append(statistics.median(ratios))
    values.sort()
    return {
        "draws":draws,
        "resampling_unit":"one entire NOR18 source cohort of 16 cases",
        "percentile_2_5":values[int(.025*(draws-1))],
        "percentile_97_5":values[int(.975*(draws-1))],
    }


def host_info():
    return {
        "python_version":sys.version,
        "python_implementation":platform.python_implementation(),
        "python_compiler":platform.python_compiler(),
        "platform":platform.platform(),
        "machine":platform.machine(),
        "processor":platform.processor(),
        "logical_cpu_count":os.cpu_count(),
        "benchmark_clock":"time.perf_counter_ns",
        "garbage_collection":"disabled during measured segment, collected before each",
        "tracemalloc":"separate untimed call for each case/mode",
        "RSS":"separate subprocess per mode; Linux ru_maxrss peak",
    }


def get_rss_kib():
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value/1024) if sys.platform=="darwin" else int(value)
    except (ImportError,AttributeError):
        return None


def rss_worker(fixture_path,mode):
    fixture=load_fixture(fixture_path)
    before=get_rss_kib()
    counter=0
    for row in fixture["cases"]:
        c,h=unpack_case(row)
        outcome(c,h,mode)
        counter+=1
    after=get_rss_kib()
    assert counter==SOURCES
    return {
        "mode":mode,
        "baseline_peak_rss_KiB_before_algorithm":before,
        "process_peak_rss_KiB_after_algorithm":after,
        "max_rss_increase_KiB":(
            max(0,after-before) if after is not None and before is not None else None),
        "cases":counter,
        "platform":platform.platform(),
        "process_exit_separate":True,
    }


def isolated_rss(fixture_path):
    out={}
    for mode in MODES:
        worker=subprocess.run([
            sys.executable,str(Path(__file__).resolve()),
            "--fixture",str(fixture_path),"--rss-worker",mode],
            check=True,text=True,capture_output=True)
        last=worker.stdout.strip().splitlines()[-1]
        x=json.loads(last)
        assert x["mode"]==mode and x["cases"]==64
        out[mode]=x
    return out


def run(fixture_path,timed_reps=TIMED_REPS,warmups=WARMUPS):
    if timed_reps!=TIMED_REPS or warmups!=WARMUPS:
        raise ValueError("registered timing repetitions or warmups changed")
    fixture=load_fixture(fixture_path)
    output=[]
    families=Counter()
    for position,row in enumerate(fixture["cases"]):
        c,h=unpack_case(row)
        families[c.family]+=1
        expected=(row["original_exact_full_accept_mask"],
                  row["original_public_future_output_identifiable"])
        # Critical algorithmic parity and a rigorous source-only baseline:
        observed={mode:outcome(c,h,mode) for mode in MODES}
        assert len({a["mask"] for a in observed.values()})==1
        assert len({a["unanimous"] for a in observed.values()})==1
        assert all((a["mask"],a["unanimous"])==expected
                   for a in observed.values())

        # Start each case with its own deterministic order schedule.
        per_mode={mode:[] for mode in MODES}
        rng=random.Random(SEED+position*100003)
        for rep in range(warmups+timed_reps):
            order=list(MODES)
            rng.shuffle(order)
            for mode in order:
                gc.collect()
                was_enabled=gc.isenabled()
                gc.disable()
                try:
                    measured=outcome(c,h,mode,measure=True)
                finally:
                    if was_enabled:gc.enable()
                assert measured["mask"]==expected[0]
                if rep>=warmups:
                    per_mode[mode].append(measured["times"])
        assert all(len(v)==TIMED_REPS for v in per_mode.values())
        medians={mode:{
            "query_ns":median_ns(per_mode[mode],"query_ns"),
            "verification_ns":median_ns(per_mode[mode],"verification_ns"),
            "certificate_ns":median_ns(per_mode[mode],"certificate_ns"),
            "total_ns":median_ns(per_mode[mode],"total_ns")}
            for mode in MODES}
        temporary={mode:ephemeral_memory_profile(c,h,mode) for mode in MODES}
        assert all(z["mask"]==expected[0] for z in temporary.values())
        short={str(degree):nor17.prefix_mask(c,h,degree)
               for degree in (160,176)}
        bigint=integer_cost_ledger(c,h,short)
        speed=medians["independent16"]["total_ns"]/max(
            1.,medians["cached16"]["total_ns"])
        output.append({
            "cohort":row["cohort"],
            "family":row["family"],
            "original_index":row["original_index"],
            "full_public_input_sha256":row["original_full_public_sha256"],
            "public_header":row["public_header"],
            "reference_full_public_accept_mask":expected[0],
            "reference_unanimous_future_output":expected[1],
            "full_accepted_template_count":expected[0].bit_count(),
            "prefix_public_survivor_masks":short,
            "algorithm_result_parity_all_modes":True,
            "measured_ns":per_mode,
            "median_ns":medians,
            "untimed_tracemalloc":temporary,
            "bigint_operand_width_ledger":bigint,
            "speedup_independent_over_cached_total":speed,
        })
    assert families=={"late_source":32,"signed":16,"rational":16}
    ratios=[x["speedup_independent_over_cached_total"] for x in output]
    median_ratio=statistics.median(ratios)
    primary_pass=(
        median_ratio>=1.5 and
        sum(x>1 for x in ratios)>=48 and
        all(x["algorithm_result_parity_all_modes"] for x in output)
    )
    def family_summary(rows):
        selected=[x for x in output if x["family"] in rows]
        if not selected:return None
        return {
            "cases":len(selected),
            "median_paired_total_speedup":statistics.median(
                x["speedup_independent_over_cached_total"] for x in selected),
            "cached_faster_cases":sum(
                x["speedup_independent_over_cached_total"]>1 for x in selected),
            "per_mode_median_total_ms":{mode:statistics.median(
                x["median_ns"][mode]["total_ns"] for x in selected)/1e6
                for mode in MODES},
            "per_mode_median_verification_ms":{mode:statistics.median(
                x["median_ns"][mode]["verification_ns"] for x in selected)/1e6
                for mode in MODES},
            "per_mode_median_temporary_tracemalloc_KiB":{mode:statistics.median(
                x["untimed_tracemalloc"][mode][
                    "python_tracemalloc_peak_bytes"] for x in selected)/1024
                for mode in MODES},
            "per_mode_total_R_integer_multiplications":{mode:sum(
                x["bigint_operand_width_ledger"]["per_mode"][mode][
                    "R_integer_multiplications_executed"] for x in selected)
                for mode in MODES},
            "per_mode_total_R_operand_width_product_proxy":{mode:sum(
                x["bigint_operand_width_ledger"]["per_mode"][mode][
                    "R_operand_bitwidth_product_proxy"] for x in selected)
                for mode in MODES},
        }
    return {
        "schema":SCHEMA,
        "original_NOR18_full_JSON_SHA256":NOR18_SHA,
        "fixture_schema":FIXTURE_SCHEMA,
        "fixture_sha256":sha256(Path(fixture_path).read_bytes()).hexdigest(),
        "fixture_cases":SOURCES,
        "source_families":dict(families),
        "timed_repetitions":TIMED_REPS,
        "untimed_warmups_per_case_per_algorithm":WARMUPS,
        "modes":list(MODES),
        "host":host_info(),
        "elapsed_setup_excluded_from_timing":True,
        "query_input_degree":QUERY_DEGREE,
        "public_demo_degree":PUBLIC_DEGREE,
        "per_case":output,
        "summary_all":family_summary(set(families)),
        "summary_by_family":{family:family_summary({family}) for family in families},
        "primary_median_paired_total_speedup":median_ratio,
        "primary_cached_faster_cases":sum(v>1 for v in ratios),
        "primary_fraction_cached_faster":sum(v>1 for v in ratios)/SOURCES,
        "primary_cohort_block_bootstrap":block_bootstrap(output),
        "primary_preregistered_speedup_threshold":1.5,
        "primary_preregistered_faster_case_threshold":48,
        "primary_preregistered_passed":primary_pass,
        "isolated_process_RSS_KiB":isolated_rss(fixture_path),
        "scope":{
            "all_source_inputs_from_original_NOR18_288_case_test":True,
            "exact_public_mask_and_certificate_match_original_for_every_mode":True,
            "future_outputs_and_private_generator_programs_not_in_fixture":True,
            "native_independent_full_verifier_is_unmodified":True,
            "cached_public_verifier_uses_existing_NOR17_compound_series":True,
            "fixed_public_query_recomputed_for_each_mode":True,
            "time_includes_python_bigint_implementation_effects":True,
            "input_serialization_and_cohort_generation_excluded_from_timing":True,
            "cohort_bootstrap_not_independent_research_replication":True,
            "reciprocal_integer_operand_width_products_are_cost_proxy":True,
            "RSS_and_tracemalloc_measured_in_separate_untimed_passes":True,
            "no_general_theorem_proving_or_GPU_speedup_claim":True,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--repetitions",type=int,default=TIMED_REPS)
    p.add_argument("--warmups",type=int,default=WARMUPS)
    p.add_argument("--rss-worker",choices=MODES)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    if args.make_fixture:
        fixture=fixture_from_locked_source()
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_text(json.dumps(
            fixture,indent=2,sort_keys=True)+"\n")
        assert len(fixture["cases"])==SOURCES
        print("NOR18R ORIGINAL SOURCE INTEGRITY AND PUBLIC-ONLY FIXTURE PASSED",
              {"cases":64,"sha256":sha256(
                  args.make_fixture.read_bytes()).hexdigest()})
        return
    if not args.fixture:
        p.error("pass --make-fixture or --fixture")
    if args.rss_worker:
        print(json.dumps(rss_worker(args.fixture,args.rss_worker),
                         sort_keys=True))
        return
    result=run(args.fixture,args.repetitions,args.warmups)
    if args.check:
        assert result["fixture_cases"]==64
        assert result["timed_repetitions"]==7
        assert result["primary_cached_faster_cases"]<=64
        assert result["primary_cohort_block_bootstrap"]["draws"]==4096
        assert len(result["per_case"])==64
        assert all(x["algorithm_result_parity_all_modes"] for x in result["per_case"])
    doc=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(doc)
    else:
        print(doc,end="")


if __name__=="__main__":
    main()
