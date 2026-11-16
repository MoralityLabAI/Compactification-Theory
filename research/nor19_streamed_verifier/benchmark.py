"""NOR-19: public-only streaming / chunked cached verifier benchmarks.

The primary implementation streams *coefficients* through the exact
RRC25 two-branch delayed-series expression, retains the four unary
transforms for one public demo at a time, and never creates any candidate
output series for demo verification.

Compare the original NOR18R native and cached algorithms against two
streamed execution orders and a bounded four-output chunk checker.
Independent fixture source is the previously UNBENCHMARKED half of the
same NOR18 frozen 288-case heldout corpus, with every source digest
rechecked against its original public first-run JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
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
from research.nor18r_runtime import benchmark as old

SCHEMA="nor19.streaming_memory_bounded_public_verifier.v1"
FIXTURE_SCHEMA="nor19.previously_untimed_public_source_64.v1"
OLD_FULL_SHA=old.NOR18_SHA
CHECKPOINT_SHA=old.CHECKPOINT_SHA
SOURCE_MODELS=("independent16","cached16","stream16_queryfirst",
               "stream16_verifyfirst","chunk4_verifyfirst")
WARMUPS=2
TIMED_REPS=7
SOURCES=64
BOOTSTRAP_DRAWS=4096
BOOT_SEED=202611191900
PRIMARY="stream16_verifyfirst"
PRIMARY_MEMORY_RATIO_MAX=.80
PRIMARY_SPEEDUP_NATIVE_MIN=2.0
PRIMARY_MEMORY_WINS_MIN=48


def json_bytes(obj):
    return (json.dumps(obj,indent=2,sort_keys=True)+"\n").encode()


def make_fixture():
    archive=old.read_nor18()
    roster=nor11.sample_cohorts(nor12.SPLIT_COUNTS)
    batches=nor18.fresh_cohorts(roster)
    assert len(batches)==4
    records=[]
    for rep,batch in enumerate(batches):
        assert len(batch)==72
        assert sha256("\n".join(nor16.full_public_digest(c)
                for c,h,d in batch).encode()).hexdigest()==archive["cohorts"][rep][
                    "public_manifest_sha256"]
        for i,(c,h,model_hash) in enumerate(batch):
            source=archive["records"][rep*72+i]
            assert source["family"]==c.family
            assert source["full_public_input_sha256"]==nor16.full_public_digest(c)
            assert source["model_visible_sha256"]==model_hash
        # Original NOR18R used 0:8, 48:52, 60:64. Keep these new.
        indices=list(range(8,16))+list(range(52,56))+list(range(64,68))
        assert len(indices)==16
        assert Counter(batch[i][0].family for i in indices)=={
            "late_source":8,"signed":4,"rational":4}
        for i in indices:
            c,h,model_hash=batch[i]
            source=archive["records"][rep*72+i]
            records.append({
                "cohort":rep,
                "original_index":rep*72+i,
                "family":c.family,
                "original_full_public_sha256":source["full_public_input_sha256"],
                "original_model_input_sha256":model_hash,
                "public_header":{"base":h.base,"delay":h.delay,
                                 "spacing":h.spacing,"amplitude":h.amplitude},
                "demos":[[list(f[:193]),list(y[:193])] for f,y in c.demos],
                "public_query_input":list(c.query),
                "original_exact_full_accept_mask":source["retro_full_accepted_mask"],
                "original_public_future_output_identifiable":source[
                    "certificate_possible_from_complete_public_mask"],
            })
    assert len(records)==SOURCES
    assert len({r["original_full_public_sha256"] for r in records})==SOURCES
    assert Counter(x["family"] for x in records)=={
        "late_source":32,"signed":16,"rational":16}
    return {
        "schema":FIXTURE_SCHEMA,
        "source_original_NOR18_sha256":OLD_FULL_SHA,
        "original_RRC25_checkpoint_sha256":CHECKPOINT_SHA,
        "selection_rule":"each of four NOR18 cohorts, indices 8:16+52:56+64:68",
        "previously_timed_NOR18R_source_indices_excluded":True,
        "public_only":True,
        "private_program_identities_and_query_outputs_absent":True,
        "cases":records,
    }


def load_fixture(path):
    raw=Path(path).read_bytes()
    o=json.loads(raw)
    assert o["schema"]==FIXTURE_SCHEMA
    assert o["source_original_NOR18_sha256"]==OLD_FULL_SHA
    assert o["original_RRC25_checkpoint_sha256"]==CHECKPOINT_SHA
    assert o["previously_timed_NOR18R_source_indices_excluded"]
    assert o["public_only"] and o["private_program_identities_and_query_outputs_absent"]
    assert len(o["cases"])==SOURCES
    assert Counter(z["family"] for z in o["cases"])=={
        "late_source":32,"signed":16,"rational":16}
    assert len({r["original_full_public_sha256"] for r in o["cases"]})==64
    return o


def public_case(row):
    # Original NOR18R loader checks source digest, feature hash and exact
    # public degree192 / query input degree256 without a secret future.
    return old.unpack_case(row)


def streamed_mask(c,h):
    """Four unary transforms cached per demo; no candidate output vectors.

    Short-circuit on the first mismatching exact integer coefficient.
    The two demos are checked sequentially so only one demo's unary
    transform workspace is live. The candidate sets form an exact
    intersection, preserving native full-template acceptance.
    """
    mask=65535
    k=h.delay
    d=h.spacing
    amp=h.amplitude
    for source,observed in c.demos:
        maps={op:corpus.d.operate(source,op) for op in corpus.OPS}
        base=maps[h.base]
        for j in range(16):
            if not mask&(1<<j):
                continue
            u=maps[corpus.OPS[j//4]]
            v=maps[corpus.OPS[j%4]]
            for n in range(corpus.OBSERVED+1):
                x=base[n]
                if n==k:
                    x+=amp
                if n>=k+d:
                    x+=amp*u[n-k-d]
                if n>=k+2*d:
                    x+=amp*v[n-k-2*d]
                if x!=observed[n]:
                    mask &= ~(1<<j)
                    break
        # Release the last candidate's references as well as the map,
        # so no preceding demo's full unary workspace overlaps the next.
        del u,v,base,maps
        if mask==0:
            break
    return mask


def build_one_candidate(h,transforms,j,degree=corpus.OBSERVED):
    """One complete exact candidate output, without an all-class batch."""
    assert 0<=j<16
    base=transforms[h.base]
    u=transforms[corpus.OPS[j//4]]
    v=transforms[corpus.OPS[j%4]]
    k=h.delay
    d=h.spacing
    amp=h.amplitude
    result=list(base)
    result[k]+=amp
    for n in range(k+d,degree+1):
        result[n]+=amp*u[n-k-d]
    for n in range(k+2*d,degree+1):
        result[n]+=amp*v[n-k-2*d]
    return tuple(result)


def chunk4_mask(c,h):
    """At most four materialized degree192 outputs, one demo at a time."""
    mask=65535
    for source,observed in c.demos:
        maps={op:corpus.d.operate(source,op) for op in corpus.OPS}
        for start in range(0,16,4):
            # No persistent output materialization across four-class chunks.
            chunk=[(j,build_one_candidate(h,maps,j))
                   for j in range(start,start+4) if mask&(1<<j)]
            for j,candidate in chunk:
                if candidate!=observed:
                    mask &=~(1<<j)
            if chunk:
                del candidate
            del chunk
        del maps
        if mask==0:
            break
    return mask


def public_mask(c,h,mode):
    if mode in ("independent16","cached16"):
        return old.evaluated_full_mask(c,h,mode)
    if mode in ("stream16_queryfirst","stream16_verifyfirst"):
        return streamed_mask(c,h)
    if mode=="chunk4_verifyfirst":
        return chunk4_mask(c,h)
    raise ValueError("unsupported public verifier mode")


def complete(c,h,mode,measure=False):
    """Same 16 public-query predictions and exact proof certificate.

    The cached original is called unchanged, retaining its original
    execution order and memory behavior. Verify-first variants release
    public verification buffers before performing query simulations.
    """
    if mode not in SOURCE_MODELS:
        raise ValueError("unregistered algorithm")
    if mode in ("independent16","cached16"):
        return old.outcome(c,h,mode,measure=measure)

    verify_first=mode.endswith("verifyfirst")
    if measure:
        start=time.perf_counter_ns()
    if verify_first:
        mask=public_mask(c,h,mode)
        if measure:
            mid=time.perf_counter_ns()
        outputs=nor17.candidate_outputs(c.query,h,corpus.FUTURE)
        if measure:
            end=time.perf_counter_ns()
            verify_ns=mid-start
            query_ns=end-mid
    else:
        outputs=nor17.candidate_outputs(c.query,h,corpus.FUTURE)
        if measure:
            mid=time.perf_counter_ns()
        mask=public_mask(c,h,mode)
        if measure:
            end=time.perf_counter_ns()
            query_ns=mid-start
            verify_ns=end-mid
    assert 0<mask<65536
    available={outputs[j] for j in range(16) if mask&(1<<j)}
    unanimous=len(available)==1
    chosen=min(j for j in range(16) if mask&(1<<j)) if unanimous else None
    if measure:
        last=time.perf_counter_ns()
        assert last>=end>=mid>=start
        times={
            "query_ns":query_ns,
            "verification_ns":verify_ns,
            "certificate_ns":last-end,
            "total_ns":last-start,
        }
        assert sum(times[k] for k in ("query_ns","verification_ns",
                                      "certificate_ns"))==times["total_ns"]
        return {"mask":mask,"unanimous":unanimous,
                "chosen_class":chosen,"times":times}
    return {"mask":mask,"unanimous":unanimous,"chosen_class":chosen}


def allocation_profile(c,h,mode):
    gc.collect()
    tracemalloc.start()
    try:
        result=complete(c,h,mode)
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "python_peak_allocation_bytes":peak,
        "python_current_allocation_bytes":current,
        "mask":result["mask"],
    }


def verifier_only_profile(c,h,mode):
    gc.collect()
    tracemalloc.start()
    try:
        mask=public_mask(c,h,mode)
        current,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"python_verifier_only_peak_bytes":peak,
            "python_verifier_only_current_bytes":current,
            "mask":mask}


def get_rss_kib():
    return old.get_rss_kib()


def rss_worker(fixture_path,mode):
    fixture=load_fixture(fixture_path)
    before=get_rss_kib()
    for row in fixture["cases"]:
        c,h=public_case(row)
        complete(c,h,mode)
    after=get_rss_kib()
    return {
        "mode":mode,
        "cases":SOURCES,
        "baseline_peak_RSS_KiB":before,
        "peak_RSS_after_all_cases_KiB":after,
        "additional_max_RSS_KiB":(max(0,after-before)
                                  if before is not None and after is not None
                                  else None),
        "separate_process":True,
    }


def rss_by_mode(fixture_path):
    results={}
    for mode in SOURCE_MODELS:
        res=subprocess.run([
            sys.executable,str(Path(__file__).resolve()),
            "--fixture",str(fixture_path),"--rss-worker",mode],
            check=True,capture_output=True,text=True)
        parsed=json.loads(res.stdout.strip().splitlines()[-1])
        assert parsed["mode"]==mode and parsed["cases"]==SOURCES
        results[mode]=parsed
    return results


def inverse_invocations(h,mode):
    if mode=="independent16":
        return 8+16*int(h.base=="R")
    assert mode in SOURCE_MODELS
    return 1


def observed_bigint_work(c,h,mode,inverses=None):
    """Exact inverse recurrence multiplicity times observed operand width.

    All methods simulate one public query inverse. The four cached modes
    compute exactly one inverse per public demonstration. Native computes
    8 or 24 inverse recurrences per demonstration.
    """
    if inverses is None:
        inverses=[old.exact_inverse_operand_ledger(f) for f,y in c.demos]
        inverses.append(old.exact_inverse_operand_ledger(c.query))
    assert len(inverses)==3
    mult=inverse_invocations(h,mode)
    q=inverses[2]
    inverse_products=sum(r["multiplications"] for r in inverses[:2])
    inverse_width=sum(r["operand_bitwidth_products"] for r in inverses[:2])
    return {
        "R_full_public_demo_invocations":2*mult,
        "R_public_query_invocations":1,
        "R_integer_products":q["multiplications"]+mult*inverse_products,
        "R_operand_bitwidth_product_proxy":q["operand_bitwidth_products"]+
                                          mult*inverse_width,
    }


def bootstrap_ratio(records,key,draws=BOOTSTRAP_DRAWS):
    groups=defaultdict(list)
    for row in records:
        groups[row["cohort"]].append(row)
    assert set(groups)=={0,1,2,3}
    assert all(len(group)==16 for group in groups.values())
    rng=random.Random(BOOT_SEED+13)
    values=[]
    for _ in range(draws):
        sample=[x for rep in (rng.randrange(4) for k in range(4))
                for x in groups[rep]]
        values.append(statistics.median(x[key] for x in sample))
    values.sort()
    return {
        "draws":draws,
        "unit":"NOR18 original source cohort of 16 previously unbenchmarked cases",
        "percentile_2_5":values[int(.025*(draws-1))],
        "percentile_97_5":values[int(.975*(draws-1))],
    }


def env_info():
    return {
        "python":sys.version,
        "implementation":platform.python_implementation(),
        "compiler":platform.python_compiler(),
        "platform":platform.platform(),
        "machine":platform.machine(),
        "cpu_logical":os.cpu_count(),
        "time_clock":"time.perf_counter_ns",
        "measurement_gc":"collect before samples, disable during sample",
        "tracemalloc":"each mode/case separate untimed profile",
        "RSS":"isolated subprocess per algorithm, whole fixture",
    }


def run(fixture_path,timed_reps=TIMED_REPS,warmups=WARMUPS):
    if timed_reps!=TIMED_REPS or warmups!=WARMUPS:
        raise ValueError("preregistered sample counts changed")
    fixture=load_fixture(fixture_path)
    observations=[]
    for position,row in enumerate(fixture["cases"]):
        c,h=public_case(row)
        expected=(row["original_exact_full_accept_mask"],
                  row["original_public_future_output_identifiable"])
        assert all((z["mask"],z["unanimous"])==expected
                   for z in (complete(c,h,mode) for mode in SOURCE_MODELS))
        samples={name:[] for name in SOURCE_MODELS}
        rng=random.Random(BOOT_SEED+position*100003)
        for rep in range(WARMUPS+TIMED_REPS):
            modes=list(SOURCE_MODELS)
            rng.shuffle(modes)
            for mode in modes:
                gc.collect()
                enabled=gc.isenabled()
                gc.disable()
                try:
                    result=complete(c,h,mode,measure=True)
                finally:
                    if enabled:
                        gc.enable()
                assert (result["mask"],result["unanimous"])==expected
                if rep>=WARMUPS:
                    samples[mode].append(result["times"])
        assert all(len(x)==TIMED_REPS for x in samples.values())
        medians={mode:{key:float(statistics.median(s[key] for s in ss))
                       for key in ("query_ns","verification_ns","certificate_ns",
                                   "total_ns")}
                 for mode,ss in samples.items()}
        allocation={mode:allocation_profile(c,h,mode) for mode in SOURCE_MODELS}
        verifier_peak={mode:verifier_only_profile(c,h,mode)
                       for mode in SOURCE_MODELS}
        assert all(z["mask"]==expected[0] for z in allocation.values())
        assert all(z["mask"]==expected[0] for z in verifier_peak.values())
        inv=[old.exact_inverse_operand_ledger(f) for f,y in c.demos]
        inv.append(old.exact_inverse_operand_ledger(c.query))
        bigint={mode:observed_bigint_work(c,h,mode,inv)
                for mode in SOURCE_MODELS}
        primary_peak=allocation[PRIMARY]["python_peak_allocation_bytes"]
        cached_peak=allocation["cached16"]["python_peak_allocation_bytes"]
        primary_time=medians[PRIMARY]["total_ns"]
        baseline_time=medians["independent16"]["total_ns"]
        observations.append({
            "cohort":row["cohort"],
            "original_index":row["original_index"],
            "family":row["family"],
            "public_sha256":row["original_full_public_sha256"],
            "source_header":row["public_header"],
            "reference_mask":expected[0],
            "reference_unanimous":expected[1],
            "exact_mask_and_certificate_parity":True,
            "mode_samples_ns":samples,
            "mode_medians_ns":medians,
            "complete_tracemalloc":allocation,
            "verifier_only_tracemalloc":verifier_peak,
            "reciprocal_bigint_work":bigint,
            "primary_peak_ratio_vs_cached":primary_peak/cached_peak,
            "primary_native_speedup":baseline_time/max(1,primary_time),
            "streaming_lower_peak_than_cached":primary_peak<cached_peak,
        })
    assert Counter(o["family"] for o in observations)=={
        "late_source":32,"signed":16,"rational":16}
    memory_ratio=statistics.median(
        o["primary_peak_ratio_vs_cached"] for o in observations)
    speedup=statistics.median(
        o["primary_native_speedup"] for o in observations)
    wins=sum(o["streaming_lower_peak_than_cached"] for o in observations)
    passed=memory_ratio<=PRIMARY_MEMORY_RATIO_MAX and (
        speedup>=PRIMARY_SPEEDUP_NATIVE_MIN) and (
        wins>=PRIMARY_MEMORY_WINS_MIN) and all(
            o["exact_mask_and_certificate_parity"] for o in observations)
    def summarize(sample):
        return {
            "cases":len(sample),
            "median_streamed_peak_vs_cached_ratio":statistics.median(
                r["primary_peak_ratio_vs_cached"] for r in sample),
            "streamed_lower_peak_cases":sum(
                r["streaming_lower_peak_than_cached"] for r in sample),
            "median_native_over_primary_streamed_total_speedup":statistics.median(
                r["primary_native_speedup"] for r in sample),
            "median_complete_ms":{mode:statistics.median(
                r["mode_medians_ns"][mode]["total_ns"] for r in sample)/1e6
                for mode in SOURCE_MODELS},
            "median_verifier_ms":{mode:statistics.median(
                r["mode_medians_ns"][mode]["verification_ns"] for r in sample)/1e6
                for mode in SOURCE_MODELS},
            "median_complete_peak_KiB":{mode:statistics.median(
                r["complete_tracemalloc"][mode]["python_peak_allocation_bytes"]
                for r in sample)/1024 for mode in SOURCE_MODELS},
            "median_verifier_only_peak_KiB":{mode:statistics.median(
                r["verifier_only_tracemalloc"][mode][
                    "python_verifier_only_peak_bytes"] for r in sample)/1024
                for mode in SOURCE_MODELS},
            "reciprocal_bigint_products":{mode:sum(
                r["reciprocal_bigint_work"][mode]["R_integer_products"]
                for r in sample) for mode in SOURCE_MODELS},
        }
    return {
        "schema":SCHEMA,
        "original_NOR18_full_SHA256":OLD_FULL_SHA,
        "fixture_schema":FIXTURE_SCHEMA,
        "fixture_sha256":sha256(Path(fixture_path).read_bytes()).hexdigest(),
        "case_count":64,
        "cohorts":4,
        "source_families":{"late_source":32,"signed":16,"rational":16},
        "measured_repetitions":TIMED_REPS,
        "warmups":WARMUPS,
        "algorithms":list(SOURCE_MODELS),
        "primary":"stream16_verifyfirst",
        "host":env_info(),
        "records":observations,
        "summary_all":summarize(observations),
        "summary_by_family":{family:summarize([
            row for row in observations if row["family"]==family])
            for family in ("late_source","signed","rational")},
        "primary_median_complete_peak_ratio_vs_cached":memory_ratio,
        "primary_median_total_native_over_stream_speedup":speedup,
        "primary_lower_peak_cases":wins,
        "primary_median_peak_ratio_threshold":PRIMARY_MEMORY_RATIO_MAX,
        "primary_faster_than_native_threshold":PRIMARY_SPEEDUP_NATIVE_MIN,
        "primary_memory_win_case_threshold":PRIMARY_MEMORY_WINS_MIN,
        "primary_passed":passed,
        "primary_memory_block_bootstrap":bootstrap_ratio(observations,
                                   "primary_peak_ratio_vs_cached"),
        "primary_speed_block_bootstrap":bootstrap_ratio(observations,
                                   "primary_native_speedup"),
        "isolated_process_RSS_KiB":rss_by_mode(fixture_path),
        "materialized_demo_candidate_buffers_at_once_design":{
            "cached16":32,"stream16_queryfirst":0,
            "stream16_verifyfirst":0,"chunk4_verifyfirst":4,
            "independent16":"native temporary per candidate"},
        "scope":{
            "complete_public_original_masks_and_certificates_unchanged":True,
            "previous_NOR18R_measured_cases_excluded":True,
            "all_fixture_sources_match_original_NOR18_byte_locked_archive":True,
            "all_five_query_simulations_recomputed_independently":True,
            "withheld_true_future_and_program_identity_absent_from_fixture":True,
            "full_measured_times_include_query_and_verification":True,
            "tracemalloc_is_separate_untimed_execution":True,
            "memory_buffer_counts_are_design_bounds_not_bytes":True,
            "block_bootstrap_is_conditional_on_four_existing_source_cohorts":True,
            "hosted_CPU_is_not_cross_platform_guarantee":True,
        },
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--make-fixture",type=Path)
    parser.add_argument("--fixture",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--repetitions",type=int,default=TIMED_REPS)
    parser.add_argument("--warmups",type=int,default=WARMUPS)
    parser.add_argument("--rss-worker",choices=SOURCE_MODELS)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.make_fixture:
        source=make_fixture()
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(json_bytes(source))
        print("NOR19 VERIFIED NEW UNTIMERD PUBLIC FIXTURE",
              len(source["cases"]),
              sha256(args.make_fixture.read_bytes()).hexdigest())
        return
    if not args.fixture:
        parser.error("pass --make-fixture or --fixture")
    if args.rss_worker:
        print(json.dumps(rss_worker(args.fixture,args.rss_worker),
                         sort_keys=True))
        return
    result=run(args.fixture,args.repetitions,args.warmups)
    if args.check:
        assert result["case_count"]==64
        assert len(result["records"])==64
        assert result["measured_repetitions"]==7
        assert all(r["exact_mask_and_certificate_parity"] for r in result["records"])
    rendered=json_bytes(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(rendered)
    else:
        print(rendered.decode(),end="")


if __name__=="__main__":
    main()
