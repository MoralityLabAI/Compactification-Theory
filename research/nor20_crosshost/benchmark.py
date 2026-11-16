"""NOR-20: deterministic wide-coefficient public sources and cross-OS CPU.

Input fixture has only public observed demonstrations and public query INPUT.
Source generator's planted RRC25 program is ephemeral and never serialized.
The two benchmark environments consume identical byte-hashed public fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor16_public_query_consensus import experiment as nor16
from research.nor18r_runtime import benchmark as nor18r
from research.nor19_streamed_verifier import benchmark as nor19

SCHEMA="nor20.wide_coefficients_cross_host_cpu_tracemalloc.v1"
FIXTURE_SCHEMA="nor20.public_only_wide_coefficients_source_v1"
SEED_OFFSET=94120000
BITS=(4,8,12,16)
FAMILIES=("wide_dense","wide_sparse","wide_late","wide_rational")
MODES=nor19.SOURCE_MODELS
PRIMARY="stream16_verifyfirst"
COHORTS=4
N_CASES=64
WARMUPS=2
TRIALS=7
BOOTSTRAPS=4096
BOOT_SEED=202611201620
NOR18_ORIGINAL_SHA="9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b"
RRC25_MODEL_SHA="54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b"


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("utf-8")


def nonzero(rng,width):
    return rng.choice((-1,1))*rng.randint(1,(1<<width)-1)


def make_source(rng,family,width,degree=corpus.FUTURE):
    """Normalized 1+q*Z[[q]] public source; width applies to sampled seeds."""
    if width not in BITS or family not in FAMILIES or degree<16:
        raise ValueError("unsupported frozen distribution")
    if family=="wide_dense":
        return (1,)+tuple(nonzero(rng,width) for _ in range(degree))
    if family=="wide_sparse":
        return (1,)+tuple(
            nonzero(rng,width) if rng.random()<.20 else 0
            for _ in range(degree))
    if family=="wide_late":
        onset=rng.randrange(57,79)
        values=[1]+[0]*degree
        values[onset]=nonzero(rng,width)
        for n in range(onset+1,degree+1):
            if rng.random()<.30:
                values[n]=nonzero(rng,width)
        return tuple(values)
    # The 'nominal' width applies to the short polynomial, while the
    # resulting rational-series inverse may have much wider coefficients.
    poly=[1]+[0]*degree
    poly[1]=nonzero(rng,width)
    for _ in range(4):
        degree_index=rng.randint(2,16)
        poly[degree_index]+=nonzero(rng,width)
    return corpus.d.operate(tuple(poly),"R")


def seed_for(rep,width_index,family_index):
    if not(0<=rep<COHORTS and 0<=width_index<4 and
           0<=family_index<4):
        raise ValueError("cohort / width / family outside frozen grid")
    return (corpus.SEED+SEED_OFFSET+1000003*rep+
            10007*width_index+421*family_index)


def full_public_hash(c):
    return nor16.full_public_digest(c)


def reference_certificate(c,h):
    """Public-only: no private true query continuation or program."""
    mask=sum(1<<j for j in range(16)
             if corpus.verify_template(c,h,corpus.OPS[j//4],corpus.OPS[j%4]))
    if not mask:
        raise AssertionError("generated positive program has no accepted class")
    predictions=nor19.nor17.candidate_outputs(c.query,h,corpus.FUTURE)
    unanimous=len({predictions[j] for j in range(16)
                   if mask&(1<<j)})==1
    return mask,unanimous


def sample_case(rep,wi,fi,seen):
    width=BITS[wi]
    family=FAMILIES[fi]
    seed=seed_for(rep,wi,fi)
    rng=random.Random(seed)
    for attempt in range(500):
        program=(rng.choice(corpus.OPS),rng.choice(corpus.DELAYS),
                 rng.choice(corpus.SPACINGS),rng.choice(corpus.AMPLITUDES),
                 rng.choice(corpus.OPS),rng.choice(corpus.OPS))
        demos=tuple((f:=make_source(rng,family,width),
                     corpus.execute(f,program)) for _ in range(2))
        query=make_source(rng,family,width)
        # The planted program remains local and never enters the case.
        c=corpus.Case(f"nor20-r{rep}-w{width}-{family}-t{attempt}",
                      family,demos,query,(),None)
        public_hash=full_public_hash(c)
        if public_hash in seen:
            continue
        headers=corpus.possible_headers(c)
        if len(headers)!=1:
            continue
        h=headers[0]
        mask,unanimity=reference_certificate(c,h)
        chosen_index=(corpus.OPS.index(program[4])*4+
                      corpus.OPS.index(program[5]))
        # Only validate planted family membership during source creation.
        assert (h.base,h.delay,h.spacing,h.amplitude)==program[:4]
        assert mask&(1<<chosen_index)
        seen.add(public_hash)
        return c,h,attempt,mask,unanimity,public_hash
    raise RuntimeError(f"source exhausted rep={rep} width={width} family={family}")


def excluded_original_hashes():
    """Check source collision with original NOR18/NOR18R/NOR19 test cases."""
    with nor18r.ZipFile(nor18r.NOR18_ZIP) as z:
        source=z.read("nor18_full.json")
    assert len(source)==12896423
    assert sha256(source).hexdigest()==NOR18_ORIGINAL_SHA
    doc=json.loads(source)
    return {x["full_public_input_sha256"] for x in doc["records"]}


def public_max_bits(c):
    f=[v for src,y in c.demos for v in src[:193]]+list(c.query)
    o=[v for src,y in c.demos for v in y[:193]]
    return max(abs(x).bit_length() for x in f),max(
        abs(x).bit_length() for x in o)


def generate_fixture():
    seen=excluded_original_hashes()
    records=[]
    for rep in range(COHORTS):
        for wi,width in enumerate(BITS):
            for fi,family in enumerate(FAMILIES):
                c,h,attempt,mask,unique,digest=sample_case(rep,wi,fi,seen)
                max_source,max_observed=public_max_bits(c)
                records.append({
                    "cohort":rep,
                    "width_bits_seed":width,
                    "family":family,
                    "public_header":{"base":h.base,"delay":h.delay,
                        "spacing":h.spacing,"amplitude":h.amplitude},
                    "source_seed":seed_for(rep,wi,fi),
                    "accepted_attempt":attempt,
                    "demos":[[list(src[:193]),list(y[:193])] for src,y in c.demos],
                    "public_query_input":list(c.query),
                    "full_public_sha256":digest,
                    "original_exact_accept_mask":mask,
                    "original_query_output_unanimity":unique,
                    "actual_max_public_source_bits":max_source,
                    "actual_max_public_output_bits":max_observed,
                })
    assert len(records)==N_CASES and len({r["full_public_sha256"]
                                           for r in records})==N_CASES
    assert Counter(r["width_bits_seed"] for r in records)=={
        b:16 for b in BITS}
    assert Counter(r["family"] for r in records)=={
        f:16 for f in FAMILIES}
    assert Counter(r["cohort"] for r in records)=={j:16 for j in range(4)}
    return {
        "schema":FIXTURE_SCHEMA,
        "source_seed_offset":SEED_OFFSET,
        "original_NOR18_SHA256_excluded":NOR18_ORIGINAL_SHA,
        "frozen_RRC25_checkpoint_sha256":RRC25_MODEL_SHA,
        "nominal_widths":list(BITS),
        "public_families":list(FAMILIES),
        "public_fixture_only":True,
        "private_program_identities_and_future_query_targets_absent":True,
        "cases":records,
    }


def open_fixture(path):
    b=Path(path).read_bytes()
    fixture=json.loads(b)
    assert fixture["schema"]==FIXTURE_SCHEMA
    assert fixture["source_seed_offset"]==SEED_OFFSET
    assert fixture["original_NOR18_SHA256_excluded"]==NOR18_ORIGINAL_SHA
    assert fixture["frozen_RRC25_checkpoint_sha256"]==RRC25_MODEL_SHA
    assert fixture["public_fixture_only"]
    assert fixture["private_program_identities_and_future_query_targets_absent"]
    assert fixture["nominal_widths"]==list(BITS)
    assert fixture["public_families"]==list(FAMILIES)
    assert len(fixture["cases"])==N_CASES
    assert Counter(z["width_bits_seed"] for z in fixture["cases"])=={
        b:16 for b in BITS}
    assert Counter(z["family"] for z in fixture["cases"])=={
        f:16 for f in FAMILIES}
    assert Counter(z["cohort"] for z in fixture["cases"])=={
        j:16 for j in range(4)}
    return fixture,sha256(b).hexdigest()


def unpack(row):
    h=row["public_header"]
    header=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    demos=tuple((tuple(x),tuple(y)) for x,y in row["demos"])
    query=tuple(row["public_query_input"])
    assert len(demos)==2 and all(len(x)==len(y)==193 for x,y in demos)
    assert len(query)==257 and query[0]==1
    c=corpus.Case(f"nor20-public-{row['cohort']}-{row['family']}",
                  row["family"],demos,query,(),None)
    assert full_public_hash(c)==row["full_public_sha256"]
    assert c.program is None and c.future==()
    assert public_max_bits(c)==(
        row["actual_max_public_source_bits"],
        row["actual_max_public_output_bits"])
    return c,header


def timing_summary(samples):
    return {k:float(statistics.median(x[k] for x in samples))
            for k in ("query_ns","verification_ns","certificate_ns","total_ns")}


def block_interval(rows,key):
    groups=defaultdict(list)
    for r in rows:
        groups[r["cohort"]].append(r)
    assert set(groups)=={0,1,2,3}
    assert all(len(g)==16 for g in groups.values())
    rng=random.Random(BOOT_SEED+31)
    vals=[]
    for _ in range(BOOTSTRAPS):
        selected=[r for rep in (rng.randrange(4) for _ in range(4))
                  for r in groups[rep]]
        vals.append(statistics.median(r[key] for r in selected))
    vals.sort()
    return {"draws":BOOTSTRAPS,"resampling_unit":"one entire 16-case width-and-family-balanced cohort",
            "p2_5":vals[int(.025*(BOOTSTRAPS-1))],
            "p97_5":vals[int(.975*(BOOTSTRAPS-1))]}


def rss_info(cases,mode):
    """Only process baseline and peak when OS exposes resource.ru_maxrss."""
    rss_before=nor19.get_rss_kib()
    for row in cases:
        c,h=unpack(row)
        nor19.complete(c,h,mode)
    rss_after=nor19.get_rss_kib()
    return {"mode":mode,"cases":len(cases),"baseline_peak_KiB":rss_before,
            "after_peak_KiB":rss_after,
            "delta_peak_KiB":(
                max(0,rss_after-rss_before)
                if rss_before is not None and rss_after is not None else None)}


def process_rss(fixture_path):
    reports={}
    for mode in MODES:
        completed=subprocess.run([
            sys.executable,str(Path(__file__).resolve()),
            "--fixture",str(fixture_path),"--rss-worker",mode],
            capture_output=True,text=True,check=True)
        data=json.loads(completed.stdout.strip().splitlines()[-1])
        assert data["mode"]==mode and data["cases"]==N_CASES
        reports[mode]=data
    return reports


def host_env(host):
    return {"host_label":host,"os":platform.platform(),
            "python":sys.version,"machine":platform.machine(),
            "processor":platform.processor(),"compiler":platform.python_compiler(),
            "cpu_count":os.cpu_count(),"clock":"time.perf_counter_ns",
            "memory_metric":"tracemalloc separately untimed; RSS optional"}


def run(fixture_path,host,warmups=WARMUPS,trials=TRIALS):
    if host not in ("linux_ubuntu_2404","windows_2025"):
        raise ValueError("unregistered second CPU environment")
    if warmups!=WARMUPS or trials!=TRIALS:
        raise ValueError("measurement protocol changed")
    fixture,fixture_sha=open_fixture(fixture_path)
    results=[]
    for i,row in enumerate(fixture["cases"]):
        c,h=unpack(row)
        truth=(row["original_exact_accept_mask"],
               row["original_query_output_unanimity"])
        # Verifies the public-only source's full mask and future
        # candidate agreement before timing all alternative algorithms.
        assert nor19.complete(c,h,"independent16")["mask"]==truth[0]
        for mode in MODES:
            z=nor19.complete(c,h,mode)
            assert (z["mask"],z["unanimous"])==truth,(mode,row["cohort"])
        samples={mode:[] for mode in MODES}
        order_rng=random.Random(BOOT_SEED+i*100003)
        for trial in range(WARMUPS+TRIALS):
            order=list(MODES)
            order_rng.shuffle(order)
            for mode in order:
                gc.collect()
                was_enabled=gc.isenabled()
                gc.disable()
                try:
                    sampled=nor19.complete(c,h,mode,measure=True)
                finally:
                    if was_enabled:
                        gc.enable()
                assert (sampled["mask"],sampled["unanimous"])==truth
                if trial>=WARMUPS:
                    samples[mode].append(sampled["times"])
        medians={mode:timing_summary(samples[mode]) for mode in MODES}
        memory={mode:nor19.allocation_profile(c,h,mode) for mode in MODES}
        memory_only={mode:nor19.verifier_only_profile(c,h,mode)
                     for mode in MODES}
        assert all(z["mask"]==truth[0] for z in memory.values())
        assert all(z["mask"]==truth[0] for z in memory_only.values())
        inverse=[nor18r.exact_inverse_operand_ledger(f) for f,y in c.demos]
        inverse.append(nor18r.exact_inverse_operand_ledger(c.query))
        width={mode:nor19.observed_bigint_work(c,h,mode,inverse)
               for mode in MODES}
        max_inverse=max(z["max_inverse_coefficient_bits"] for z in inverse)
        ratio=memory[PRIMARY]["python_peak_allocation_bytes"]/memory[
            "cached16"]["python_peak_allocation_bytes"]
        speed=medians["independent16"]["total_ns"]/max(1,medians[
            PRIMARY]["total_ns"])
        results.append({
            "cohort":row["cohort"],
            "nominal_input_width":row["width_bits_seed"],
            "family":row["family"],
            "public_source_sha256":row["full_public_sha256"],
            "reference_mask":truth[0],
            "reference_unanimous":truth[1],
            "mask_and_certificate_parity":True,
            "actual_public_source_bits":row["actual_max_public_source_bits"],
            "actual_public_output_bits":row["actual_max_public_output_bits"],
            "max_public_reciprocal_bits":max_inverse,
            "samples_ns":samples,
            "median_ns":medians,
            "tracemalloc_complete":memory,
            "tracemalloc_verifier_only":memory_only,
            "inverse_ledger":width,
            "primary_stream_to_cached_peak_ratio":ratio,
            "primary_native_to_stream_cpu_ratio":speed,
            "primary_memory_win":ratio<1,
        })
    def summary(rows):
        return {
            "cases":len(rows),
            "median_peak_ratio_stream_over_cache":statistics.median(
                z["primary_stream_to_cached_peak_ratio"] for z in rows),
            "median_native_to_stream_cpu_ratio":statistics.median(
                z["primary_native_to_stream_cpu_ratio"] for z in rows),
            "stream_lower_python_peak_count":sum(
                z["primary_memory_win"] for z in rows),
            "median_actual_public_source_bits":statistics.median(
                z["actual_public_source_bits"] for z in rows),
            "median_max_reciprocal_bits":statistics.median(
                z["max_public_reciprocal_bits"] for z in rows),
            "median_complete_ms":{mode:statistics.median(
                z["median_ns"][mode]["total_ns"] for z in rows)/1e6
                for mode in MODES},
            "median_python_peak_KiB":{mode:statistics.median(
                z["tracemalloc_complete"][mode]["python_peak_allocation_bytes"]
                for z in rows)/1024 for mode in MODES},
            "median_verifier_peak_KiB":{mode:statistics.median(
                z["tracemalloc_verifier_only"][mode][
                    "python_verifier_only_peak_bytes"]
                for z in rows)/1024 for mode in MODES},
        }
    overall=summary(results)
    success=(overall["median_peak_ratio_stream_over_cache"]<=.80 and
             overall["stream_lower_python_peak_count"]>=48 and
             overall["median_native_to_stream_cpu_ratio"]>=2 and
             all(z["mask_and_certificate_parity"] for z in results))
    return {
        "schema":SCHEMA,
        "fixture_schema":FIXTURE_SCHEMA,
        "fixture_sha256":fixture_sha,
        "source_seed_offset":SEED_OFFSET,
        "nominal_input_widths":list(BITS),
        "public_source_families":list(FAMILIES),
        "cases":N_CASES,"cohorts":COHORTS,
        "trials":TRIALS,"warmups":WARMUPS,
        "modes":list(MODES),"primary_mode":PRIMARY,
        "host":host_env(host),
        "records":results,
        "overall":overall,
        "by_width":{str(w):summary([z for z in results
                    if z["nominal_input_width"]==w]) for w in BITS},
        "by_family":{family:summary([z for z in results
                     if z["family"]==family]) for family in FAMILIES},
        "primary_pass":success,
        "primary_memory_block_95":block_interval(
            results,"primary_stream_to_cached_peak_ratio"),
        "primary_cpu_block_95":block_interval(
            results,"primary_native_to_stream_cpu_ratio"),
        "rss_separate_process":process_rss(fixture_path),
        "scope":{
            "identical_public_fixture_for_two_hosts":True,
            "original_RRC25_public_masks_all_cases":True,
            "private_future_and_program_identity_absent":True,
            "tracemalloc_memory_separately_untimed":True,
            "same_first_host_cpu_measurement_not_assumed":True,
            "all_five_public_query_simulations_charged":True,
            "public_source_generation_excluded_from_timed_phase":True,
            "cross_host_microarchitectures_not_guaranteed_distinct":True,
            "block_bootstrap_conditions_on_four_cohorts":True,
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--make-fixture",type=Path)
    p.add_argument("--fixture",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--host-label",choices=("linux_ubuntu_2404","windows_2025"))
    p.add_argument("--warmups",type=int,default=WARMUPS)
    p.add_argument("--trials",type=int,default=TRIALS)
    p.add_argument("--rss-worker",choices=MODES)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    if args.make_fixture:
        fixture=generate_fixture()
        args.make_fixture.parent.mkdir(parents=True,exist_ok=True)
        args.make_fixture.write_bytes(canonical(fixture))
        print("NOR20 NEW WIDE PUBLIC FIXTURE",len(fixture["cases"]),
              sha256(args.make_fixture.read_bytes()).hexdigest())
        return
    if not args.fixture:
        p.error("pass --make-fixture or --fixture")
    if args.rss_worker:
        print(json.dumps(rss_info(open_fixture(args.fixture)[0]["cases"],
                                  args.rss_worker),sort_keys=True))
        return
    if not args.host_label:
        p.error("must specify --host-label")
    obj=run(args.fixture,args.host_label,args.warmups,args.trials)
    if args.check:
        assert obj["cases"]==64 and len(obj["records"])==64
        assert obj["trials"]==7 and obj["warmups"]==2
        assert len({r["public_source_sha256"] for r in obj["records"]})==64
        assert all(r["mask_and_certificate_parity"] for r in obj["records"])
    data=canonical(obj)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(data)
    else:
        print(data.decode(),end="")


if __name__=="__main__":
    main()
