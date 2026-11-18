"""NOR50 first five-repeat 64-source CPU memoization mechanism benchmark.

Requires a separately SHA-frozen public-only source artifact from an upstream
CI job. No source-specific holdout timings may influence the arm definitions
or five original gates from REGISTRATION.md.
"""
from __future__ import annotations
import argparse,json,os,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor45_authorized_program import historical_smoke as older
from research.nor46_exact_integer_trace import trace_producer as producer
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor48_program_dispatch import dispatcher as nor48
from research.nor48_program_dispatch import benchmark_cpu as previous
from research.nor49_reciprocal_count_dispatch import selector as nor49
from research.nor50_memoized_exact import cached,cohort

REPEATS=5
ARMS=("nor45_original_direct","nor50_memoized_direct",
      "nor47_gmp_full_trace","nor49_count_R_dispatch")
FIRST_REGISTERED="a3b7a211e312235f8e9363612aff4173f50a3a4e"

def canonical(x):
    return (json.dumps(x,sort_keys=True,ensure_ascii=True,indent=2)+"\n").encode("ascii")

def arm(name,program,query,degree,untrusted_trace):
    if name==ARMS[0]:
        values=native.execute_authorized_program(program,query,degree)
        root=native.independently_computed_root(values)
    elif name==ARMS[1]:
        r=cached.exact_output_with_root(program,query,degree)
        values,root=r["output"],r["merkle_root"]
    elif name==ARMS[2]:
        values=nor48.verified_gmp_trace_values(
            untrusted_trace,query,program,degree,convolve=gmp.full_gmp)
        root=native.independently_computed_root(values)
    elif name==ARMS[3]:
        r=nor49.verify_math(program,query,degree,untrusted_trace)
        values,root=r["output"],r["merkle_root"]
    else:
        raise ValueError("unknown precommitted arm")
    return values,root

def summary(rows):
    if not rows:return {"count":0}
    return {
        "count":len(rows),
        "median_cached_over_original":statistics.median(
            r["cached_over_original"] for r in rows),
        "median_gmp_over_cached":statistics.median(
            r["gmp_over_cached"] for r in rows),
        "median_nor49_over_original":statistics.median(
            r["nor49_over_original"] for r in rows),
        "source_median_ms":{a:statistics.median(r["arm_medians_ms"][a] for r in rows)
                            for a in ARMS}
    }

def workload_blocks(rows):
    totals={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    by_block=defaultdict(list)
    for r in rows:by_block[r["replicate"]].append(r)
    blocks=sorted(by_block)
    rng=random.Random(202611500092)
    out={}
    for arm in ARMS[1:]:
        ratios=[]
        for _ in range(8192):
            sample=[x for key in (rng.choice(blocks) for i in blocks)
                    for x in by_block[key]]
            ratios.append(sum(r["arm_medians_ms"][arm] for r in sample)/
                          sum(r["arm_medians_ms"][ARMS[0]] for r in sample))
        ratios.sort()
        out[arm]={"observed_total_ratio":totals[arm]/totals[ARMS[0]],
                  "descriptive_95pct_16block_bootstrap":[ratios[204],ratios[7987]]}
    return {"sum_source_medians_ms":totals,"secondary_ratios":out,
            "resamples":8192,"seed":202611500092}

def run(public_fixture:Path,repeats=REPEATS):
    if repeats!=REPEATS:raise ValueError("registered first NOR50 repeat count fixed at 5")
    fixture,source_sha=cohort.read_fixture(public_fixture)
    assert len(fixture["cases"])==64
    assert fixture["registration_first_commit"]==FIRST_REGISTERED
    global_before=nor46.exact_convolution_prefix
    rows=[]
    for ordinal,source in enumerate(fixture["cases"]):
        d=source["degree"]
        sha=source["full_public_input_sha256"]
        proposal=nor42.nor37.public_proposal(source)
        clean,hidden,beta,hidden_op=cohort.restore_worlds(source)
        assert tuple(proposal["pred"])==clean
        program=older.spec_from_public_proposal(proposal,d,beta,hidden_op)
        query=tuple(source["public_query_input"])
        count=nor49.reciprocal_count(program,d,len(query))
        expected_root=native.independently_computed_root(hidden)
        start=time.perf_counter_ns()
        witness=producer.transcript(program,query,d)
        transcript_ms=(time.perf_counter_ns()-start)/1e6
        transcript_bytes=producer.trace_json_bytes(witness)
        # Independently verify all modes before timed repetitions.
        for name in ARMS:
            values,root=arm(name,program,query,d,witness)
            assert values==hidden and root==expected_root
        samples={name:[] for name in ARMS}
        for repetition in range(REPEATS):
            order=list(ARMS)
            random.Random(previous.old.trial_seed(
                "NOR50/INTERLEAVED",sha,repetition)).shuffle(order)
            for name in order:
                begun=time.perf_counter_ns()
                got,root=arm(name,program,query,d,witness)
                elapsed=(time.perf_counter_ns()-begun)/1e6
                assert got==hidden and root==expected_root
                assert nor46.exact_convolution_prefix is global_before
                samples[name].append(elapsed)
        med={a:statistics.median(samples[a]) for a in ARMS}
        rows.append({
            "source_sha256":sha,"replicate":source["pair_replicate"],
            "degree":d,"family":source["family"],
            "authorized_active_R_count":count,
            "unique_active_ops":sorted(set(cached.active_ops(program,d,len(query)))),
            "authorized_signed_program_ops":{
                "base":program["base"],"u":program["u"],"v":program["v"],
                "hidden_beta_nonzero":bool(program["hidden_beta"]),
                "hidden_transform":program["hidden_transform"]},
            "all_exact_outputs_and_merkle_roots_identical":True,
            "exact_root_sha256":expected_root,
            "outside_arm_transcript_production_ms":transcript_ms,
            "outside_arm_full_trace_bytes":transcript_bytes,
            "arm_medians_ms":med,"five_interleaved_samples_ms":samples,
            "cached_over_original":med[ARMS[1]]/med[ARMS[0]],
            "gmp_over_cached":med[ARMS[2]]/med[ARMS[1]],
            "nor49_over_original":med[ARMS[3]]/med[ARMS[0]]
        })
        print("NOR50 FIRST CPU",ordinal+1,"/64 active-R",count,
              "cached/direct",round(rows[-1]["cached_over_original"],4),
              "GMP/cached",round(rows[-1]["gmp_over_cached"],4),flush=True)
    assert len(rows)==64
    assert Counter(r["replicate"] for r in rows)=={rep:4 for rep in cohort.REPS}
    assert all(r["all_exact_outputs_and_merkle_roots_identical"] for r in rows)
    assert nor46.exact_convolution_prefix is global_before
    multi=[r for r in rows if r["authorized_active_R_count"]>=2]
    low=[r for r in rows if r["authorized_active_R_count"]<=1]
    sm,sl=summary(multi),summary(low)
    gates={
        "multi_R_cached_over_original_under_0p70_at_least_8":(
            len(multi)>=8 and sm["median_cached_over_original"]<.70),
        "multi_R_gmp_over_cached_under_0p90_at_least_8":(
            len(multi)>=8 and sm["median_gmp_over_cached"]<.90),
        "low_R_cached_over_original_under_1p10_at_least_8":(
            len(low)>=8 and sl["median_cached_over_original"]<1.10),
        "all_full_exact_outputs_roots_and_training_controls":(
            all(r["all_exact_outputs_and_merkle_roots_identical"] for r in rows)
            and all(len(r["unique_active_ops"])==len(set(r["unique_active_ops"]))
                    for r in rows)),
        "source_first_64_reentrant_five_interleaved_repeats":(
            len(set(r["source_sha256"] for r in rows))==64 and
            len({r["replicate"] for r in rows})==16 and
            all(all(len(r["five_interleaved_samples_ms"][a])==5 for a in ARMS)
                for r in rows) and
            nor46.exact_convolution_prefix is global_before)
    }
    return {
        "schema":"nor50.first_64_new_source_memoization_cpu.v1",
        "registration_first_commit":FIRST_REGISTERED,
        "source_only_first_sha256":source_sha,
        "source_count":64,"historical_excluded_source_count":452,
        "repeat_count":REPEATS,
        "first_five_interleaved_arms":list(ARMS),
        "gmp_backend":gmp.backend_metadata(),
        "cpu_model":previous.cpu_model(),"python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "source_only_precedes_first_timing":True,
        "no_global_NOR46_convolution_override":True,
        "common_authentication_transport_cost_excluded_equally":True,
        "transcript_production_measured_separately":True,
        "overall":summary(rows),
        "by_R_count":{str(n):summary([r for r in rows
                     if r["authorized_active_R_count"]==n]) for n in range(5)},
        "by_R_stratum":{"at_least_two":sm,"at_most_one":sl},
        "by_degree":{str(d):summary([r for r in rows if r["degree"]==d])
                     for d in cohort.DEGREES},
        "by_family":{fam:summary([r for r in rows if r["family"]==fam])
                     for fam in cohort.FAMILIES},
        "secondary_workload_totals_and_bootstrap":workload_blocks(rows),
        "five_first_registered_gates":gates,
        "first_engineering_pass":all(gates.values()),
        "source_records":rows,
        "limitations":"CPU only; signed exact program semantics with synthetic provider and full nonsuccinct trace; secret honesty external."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-fixture",required=True,type=Path)
    p.add_argument("--repeats",default=REPEATS,type=int)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    result=run(a.public_fixture,a.repeats)
    raw=canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR50 FIRST CPU ORIGINAL BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR50 FIRST ENGINEERING GATES",result["five_first_registered_gates"])
    print("NOR50 FIRST ENGINEERING PASS",result["first_engineering_pass"])

if __name__=="__main__":main()
