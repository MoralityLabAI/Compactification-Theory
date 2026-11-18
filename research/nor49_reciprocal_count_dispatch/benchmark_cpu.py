"""NOR49 first timing-clean 48-source signed-native exact backend benchmark.

Five interleaved repeats of NOR45 direct, NOR47 reentrant GMP full trace,
NOR48 signed-program any-R, NOR49 frozen R-count>=2, all with full output
and Merkle root. An immutable public-only roster is frozen before timing.
No test-source timings may influence the signed-program-only route.
"""
from __future__ import annotations
import argparse,json,os,random,statistics,sys,time
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor48_program_dispatch import benchmark_cpu as prior
from research.nor48_program_dispatch import dispatcher as old
from research.nor49_reciprocal_count_dispatch import cohort,selector
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor45_authorized_program import historical_smoke as older
from research.nor46_exact_integer_trace import trace_producer as producer
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp

REPEATS=5
ARMS=("nor45_direct","nor47_gmp_full_trace",
      "nor48_any_R_dispatch","nor49_multiple_R_dispatch")
REGISTERED="26b70716ec3d3830031919e593d3092cc31ae162"


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def check_roster(path):
    buf=Path(path).read_bytes()
    roster=json.loads(buf)
    assert roster["schema"]=="nor49.first_fixed_timing_clean_nor42_source_roster.v1"
    assert roster["first_registration_sha"]==REGISTERED
    assert roster["nor42_first_public_source_sha256"]==cohort.ORIGINAL_NOR42_SOURCE_SHA
    assert roster["nor48_first_cpu_train_sha256"]==cohort.ORIGINAL_NOR48_FIRST_CPU_SHA
    assert roster["training_source_count"]==16
    assert roster["timing_clean_source_count"]==48
    assert roster["excluded_training_replicates"]==list(cohort.TRAIN_REPS)
    assert roster["selected_test_replicates"]==list(cohort.TEST_REPS)
    assert roster["no_new_source_identity_created"] and roster["no_holdout_timing_observed"]
    assert roster["selection_contains_no_program_or_private_truth"]
    assert len(roster["cases"])==48
    ids=[r["source_sha256"] for r in roster["cases"]]
    assert len(set(ids))==48
    assert Counter(r["replicate"] for r in roster["cases"])=={
        rep:4 for rep in cohort.TEST_REPS}
    # Recheck the original frozen source roster without timing anything.
    expected=cohort.freeze_roster()
    assert expected==roster
    return roster,sha256(buf).hexdigest()


def arm(arm_name,program,query,d,untrusted_trace):
    if arm_name=="nor45_direct":
        values=native.execute_authorized_program(program,query,d)
        root=native.independently_computed_root(values)
    elif arm_name=="nor47_gmp_full_trace":
        values=old.verified_gmp_trace_values(
            untrusted_trace,query,program,d,convolve=gmp.full_gmp)
        root=native.independently_computed_root(values)
    elif arm_name=="nor48_any_R_dispatch":
        result=old.dispatch_verified_arithmetic(program,query,d,untrusted_trace)
        values,root=result["output"],result["merkle_root"]
    elif arm_name=="nor49_multiple_R_dispatch":
        result=selector.verify_math(program,query,d,untrusted_trace)
        values,root=result["output"],result["merkle_root"]
    else:raise ValueError("unknown frozen timed arm")
    return values,root


def run(roster_path,repeats=REPEATS):
    if repeats!=REPEATS:raise ValueError("first NOR49 requires five repeats")
    roster,roster_sha=check_roster(roster_path)
    fixture,graded=older.load_original()
    index={r["full_public_input_sha256"]:(r,rec)
           for r,rec in zip(fixture["cases"],graded["source_records"],strict=True)}
    assert len(index)==64
    prior_source_sha=cohort.nor48.source_provenance()
    assert set(prior_source_sha).isdisjoint(
        r["source_sha256"] for r in roster["cases"])
    saved_global=prior.nor46.exact_convolution_prefix
    results=[]
    for ordinal,source_record in enumerate(roster["cases"]):
        sid=source_record["source_sha256"]
        row,graded_record=index[sid]
        assert graded_record["public_sha256"]==sid
        proposal=prior.old.nor42.nor37.public_proposal(row)
        clean=proposal["pred"]
        _,hidden,beta,hidden_op=prior.old.nor42.restore_worlds(row,clean)
        program=older.spec_from_public_proposal(
            proposal,row["degree"],beta,hidden_op)
        query=tuple(row["public_query_input"])
        m=len(query)
        count=selector.reciprocal_count(program,row["degree"],m)
        route=selector.authorized_backend(program,row["degree"],m)
        assert route==("gmp_full_trace" if count>=2 else "nor45_direct")
        trace=producer.transcript(program,query,row["degree"])
        exact_root=native.independently_computed_root(hidden)
        for mode in ARMS:
            values,root=arm(mode,program,query,row["degree"],trace)
            assert values==hidden and root==exact_root
        timing={mode:[] for mode in ARMS}
        for repeat in range(repeats):
            order=list(ARMS)
            random.Random(prior.old.trial_seed(
                "NOR49/INTERLEAVED",sid,repeat)).shuffle(order)
            for mode in order:
                begin=time.perf_counter_ns()
                vals,root=arm(mode,program,query,row["degree"],trace)
                delta=(time.perf_counter_ns()-begin)/1e6
                assert vals==hidden and root==exact_root
                assert prior.nor46.exact_convolution_prefix is saved_global
                timing[mode].append(delta)
        med={mode:statistics.median(v) for mode,v in timing.items()}
        r_direct=med["nor49_multiple_R_dispatch"]/med["nor45_direct"]
        r_previous=med["nor49_multiple_R_dispatch"]/med["nor48_any_R_dispatch"]
        result={
            "source_sha256":sid,
            "replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "active_R_count_from_authorized_program":count,
            "program_backend":route,
            "signed_program_ops":{
                "base":program["base"],"u":program["u"],"v":program["v"],
                "hidden_transform":program["hidden_transform"],
                "hidden_beta_nonzero":program["hidden_beta"]!=0
            },
            "complete_true_future_root_sha256":exact_root,
            "full_trace_bytes":producer.trace_json_bytes(trace),
            "all_exact_outputs_roots_identical":True,
            "arm_median_ms":med,
            "interleaved_five_sample_ms":timing,
            "nor49_over_direct":r_direct,
            "nor49_over_nor48":r_previous,
            "nor48_over_direct":med["nor48_any_R_dispatch"]/med["nor45_direct"],
            "full_GMP_over_direct":med["nor47_gmp_full_trace"]/med["nor45_direct"],
        }
        results.append(result)
        print("NOR49 FIRST CLEAN",ordinal+1,"/48 R count",count,
              "new/direct",round(r_direct,4),"new/old",round(r_previous,4),
              flush=True)
    assert len(results)==48
    assert Counter(r["replicate"] for r in results)=={
        rep:4 for rep in cohort.TEST_REPS}
    assert all(r["all_exact_outputs_roots_identical"] for r in results)
    assert prior.nor46.exact_convolution_prefix is saved_global
    def summary(rows):
        if not rows:return {"count":0}
        return {
            "count":len(rows),
            "median_nor49_over_direct":statistics.median(
                r["nor49_over_direct"] for r in rows),
            "median_nor49_over_nor48":statistics.median(
                r["nor49_over_nor48"] for r in rows),
            "median_nor48_over_direct":statistics.median(
                r["nor48_over_direct"] for r in rows),
            "median_source_direct_ms":statistics.median(
                r["arm_median_ms"]["nor45_direct"] for r in rows),
            "median_source_nor49_ms":statistics.median(
                r["arm_median_ms"]["nor49_multiple_R_dispatch"] for r in rows)
        }
    s=summary(results)
    gates={
        "median_nor49_over_direct_less_than_0p75":s["median_nor49_over_direct"]<.75,
        "median_nor49_over_nor48_less_than_0p95":s["median_nor49_over_nor48"]<.95,
        "signed_program_r_count_only_routing":all(
            r["program_backend"]==("gmp_full_trace"
                if r["active_R_count_from_authorized_program"]>=2
                else "nor45_direct") for r in results),
        "all_exact_source_program_and_roots":all(
            r["all_exact_outputs_roots_identical"] for r in results),
        "first_source_freeze_reentrant_and_five_repeats":all(
            len(r["interleaved_five_sample_ms"])==4 and
            all(len(t)==5 for t in r["interleaved_five_sample_ms"].values())
            for r in results) and prior.nor46.exact_convolution_prefix is saved_global
    }
    try:
        import resource
        maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:maxrss=None
    return {
        "schema":"nor49.first_timing_clean_48_source_r_count_dispatch_cpu.v1",
        "registration_first_commit":REGISTERED,
        "design_issue":253,
        "timing_clean_first_roster_sha256":roster_sha,
        "old_first_NOR48_training_sha256":cohort.ORIGINAL_NOR48_FIRST_CPU_SHA,
        "source_pairs":48,
        "new_source_identities":0,
        "timing_training_replicates":list(cohort.TRAIN_REPS),
        "heldout_replicates":list(cohort.TEST_REPS),
        "five_repeat_interleaved_arms":list(ARMS),
        "repeat_count":repeats,
        "backend_rule":"active authorized signed program R count >= 2 chooses GMP trace",
        "first_holdout_timing_not_used_to_choose_rule":True,
        "full_exact_output_and_merkle_root_included_in_all_arm_timings":True,
        "shared_crypto_session_transport_excluded_all_arm_timings":True,
        "no_global_convolution_override":True,
        "gmp_backend":gmp.backend_metadata(),
        "cpu_model":prior.cpu_model(),"python_version":sys.version.split()[0],
        "logical_cpus":os.cpu_count(),"process_max_rss_kib":maxrss,
        "overall":s,
        "by_R_count":{str(n):summary([
            r for r in results if r["active_R_count_from_authorized_program"]==n])
            for n in range(5)},
        "by_degree":{str(d):summary([
            r for r in results if r["degree"]==d]) for d in (512,1024)},
        "by_family":{f:summary([
            r for r in results if r["family"]==f])
            for f in ("wide_dense_small","wide_regime_blocks")},
        "first_engineering_gates":gates,
        "first_engineering_pass":all(gates.values()),
        "source_records":results,
        "limitations":"Historical source identities with timing-clean holdout only, "
           "no new scientific model evaluation or cryptographic proof claim. "
           "Provider honesty and correct signed program remain external."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--roster",required=True,type=Path)
    p.add_argument("--repeats",default=5,type=int)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    result=run(a.roster,a.repeats)
    raw=canonical(result)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_bytes(raw)
    print("NOR49 FIRST HOLDOUT CPU BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR49 ORIGINAL ENGINEERING GATES",result["first_engineering_gates"])
    print("NOR49 MEDIAN SPEED RATIOS",result["overall"])


if __name__=="__main__":main()
