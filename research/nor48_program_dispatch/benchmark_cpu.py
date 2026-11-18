"""NOR-48: first fixed source-locked 5-repeat R-aware exact CPU benchmark.

Four historical verifier arms use the same original NOR46 16 signed source
programs and complete exact output Merkle roots. Session, provider and
signature transport costs are excluded equally from this arithmetic+root
comparator and tested in the NOR45/46 signed security fixtures.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor47_gmp_exact_convolution import benchmark_cpu as old
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor46_exact_integer_trace import trace_producer as producer
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor48_program_dispatch import dispatcher

SOURCE_SHA="809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530"
SOURCE_BYTES=34928
REPEATS=5
ARMS=("nor45_direct","nor46_python_full_trace","nor47_gmp_full_trace",
      "nor48_r_aware_dispatch")
ISSUE=251


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def source_provenance():
    original=ROOT/"research/nor47_gmp_exact_convolution/results/nor47_first_original_cpu.json"
    raw=original.read_bytes()
    assert len(raw)==SOURCE_BYTES and sha256(raw).hexdigest()==SOURCE_SHA
    data=json.loads(raw)
    assert data["schema"]=="nor47.first_archived_gmp_exact_convolution_cpu.v1"
    assert data["historical_source_pairs"]==16
    assert data["source_selection"]==[100,104,108,112]
    assert data["fixed_repeat_count_per_arm"]==REPEATS
    assert data["all_first_source_parity"]
    prior={r["source_sha256"]:r for r in data["source_records"]}
    assert len(prior)==16
    return prior


def cpu_model():
    return old.cpu_model()


def execute_arm(arm,program,query,degree,trace):
    if arm=="nor45_direct":
        values=nor45.execute_authorized_program(program,query,degree)
        root=nor45.independently_computed_root(values)
    elif arm=="nor46_python_full_trace":
        values=nor46.verified_trace_values(trace,query,program,degree)
        root=nor45.independently_computed_root(values)
    elif arm=="nor47_gmp_full_trace":
        # Explicit reentrant dependency injection, no global monkeypatch.
        values=dispatcher.verified_gmp_trace_values(
            trace,query,program,degree,convolve=gmp.full_gmp)
        root=nor45.independently_computed_root(values)
    elif arm=="nor48_r_aware_dispatch":
        result=dispatcher.dispatch_verified_arithmetic(
            program,query,degree,trace)
        values,root=result["output"],result["merkle_root"]
    else:
        raise ValueError("unknown benchmark arm")
    return values,root


def run(repeats=REPEATS):
    if repeats!=REPEATS:
        raise ValueError("original five-repeat protocol fixed before data")
    prior=source_provenance()
    original=old.archive.audit()
    assert original["old_source_pairs"]==16
    fixture,grade=old.nor45.load_original()
    cohort=[(s,g) for s,g in zip(
        fixture["cases"],grade["source_records"],strict=True)
        if s["full_public_input_sha256"] in prior]
    assert len(cohort)==16
    assert Counter(r["pair_replicate"] for r,_ in cohort)=={
        100:4,104:4,108:4,112:4}
    original_roots={
        x["source_sha256"]:x["signed_program_root_sha256"]
        for x in json.loads(
        (ROOT/"research/nor46_exact_integer_trace/results/nor46_first_receipt.json"
         ).read_bytes())["source_records"]}
    assert len(original_roots)==16
    baseline_global=nor46.exact_convolution_prefix
    results=[]
    for ix,(source,graded) in enumerate(cohort):
        sha=source["full_public_input_sha256"]
        assert graded["public_sha256"]==sha
        public=old.nor42.nor37.public_proposal(source)
        clean=public["pred"]
        _,hidden,beta,hiddenop=old.nor42.restore_worlds(source,clean)
        program=old.nor45.spec_from_public_proposal(
            public,source["degree"],beta,hiddenop)
        query=tuple(source["public_query_input"])
        statement_route=dispatcher.authorized_routing_key(
            program,source["degree"],len(query))
        trace=producer.transcript(program,query,source["degree"])
        assert nor46.verified_trace_values(
            trace,query,program,source["degree"])==hidden
        root=nor45.independently_computed_root(hidden)
        assert root==original_roots[sha]
        assert producer.trace_json_bytes(trace)==prior[sha]["full_trace_bytes"]
        assert (statement_route=="gmp_full_trace")==prior[sha]["has_reciprocal_R"]
        timings={p:[] for p in ARMS}
        for repetition in range(repeats):
            order=list(ARMS)
            random.Random(old.trial_seed("NOR48/ARM-ORDER",sha,repetition)).shuffle(order)
            for arm in order:
                start=time.perf_counter_ns()
                got,merkle=execute_arm(arm,program,query,source["degree"],trace)
                ms=(time.perf_counter_ns()-start)/1e6
                assert got==hidden and merkle==root
                assert nor46.exact_convolution_prefix is baseline_global
                timings[arm].append(ms)
        med={arm:statistics.median(t) for arm,t in timings.items()}
        ratio=med["nor48_r_aware_dispatch"]/med["nor45_direct"]
        results.append({
            "source_sha256":sha,
            "replicate":source["pair_replicate"],
            "degree":source["degree"],"family":source["family"],
            "has_R_in_authorized_program":statement_route=="gmp_full_trace",
            "authorized_program_routing":statement_route,
            "nor47_original_has_R":prior[sha]["has_reciprocal_R"],
            "signed_program_root_sha256":root,
            "all_four_full_exact_outputs_and_roots_equal":True,
            "original_full_trace_bytes":producer.trace_json_bytes(trace),
            "arm_medians_ms":med,
            "five_interleaved_arm_ms":timings,
            "dispatcher_over_direct":ratio,
            "gmp_over_direct":med["nor47_gmp_full_trace"]/med["nor45_direct"],
            "dispatcher_over_gmp":med["nor48_r_aware_dispatch"]/
                                  med["nor47_gmp_full_trace"],
            "direct_selected_for_no_R":statement_route=="nor45_direct" and
                                      not prior[sha]["has_reciprocal_R"],
        })
        print("NOR48 historical",ix+1,"/16 R",
              results[-1]["has_R_in_authorized_program"],
              "dispatch/direct",round(ratio,4),flush=True)
    assert len(results)==16
    assert Counter(r["has_R_in_authorized_program"] for r in results)=={
        True:12,False:4}
    assert all(r["all_four_full_exact_outputs_and_roots_equal"] for r in results)
    assert all(r["direct_selected_for_no_R"] for r in results if not
               r["has_R_in_authorized_program"])
    assert nor46.exact_convolution_prefix is baseline_global
    def summary(rows):
        return {
            "source_count":len(rows),
            "median_of_sourcewise_dispatch_over_direct":statistics.median(
                r["dispatcher_over_direct"] for r in rows),
            "median_of_sourcewise_gmp_over_direct":statistics.median(
                r["gmp_over_direct"] for r in rows),
            "median_of_sourcewise_dispatch_over_gmp":statistics.median(
                r["dispatcher_over_gmp"] for r in rows),
            "direct_median_ms":statistics.median(
                r["arm_medians_ms"]["nor45_direct"] for r in rows),
            "dispatch_median_ms":statistics.median(
                r["arm_medians_ms"]["nor48_r_aware_dispatch"] for r in rows),
        }
    groups={name:summary([r for r in results if r["has_R_in_authorized_program"]==flag])
            for name,flag in (("R",True),("no_R",False))}
    overall=summary(results)
    gates={
        "dispatcher_median_ratio_under_0p75":overall[
            "median_of_sourcewise_dispatch_over_direct"]<.75,
        "all_four_no_R_use_direct":sum(r["direct_selected_for_no_R"] for r in results)==4,
        "all_signed_exact_source_roots_identical":all(
            r["all_four_full_exact_outputs_and_roots_equal"] for r in results),
        "all_R_use_injected_GMP":all(
            r["authorized_program_routing"]=="gmp_full_trace"
            for r in results if r["has_R_in_authorized_program"]),
        "global_NOR46_convolution_function_unchanged":(
            nor46.exact_convolution_prefix is baseline_global)
    }
    return {
        "schema":"nor48.first_source_locked_signed_program_r_dispatch_cpu.v1",
        "design_issue":ISSUE,
        "nor47_first_cpu_bytes":SOURCE_BYTES,
        "nor47_first_cpu_sha256":SOURCE_SHA,
        "source_pairs":16,
        "registered_replicate_blocks":[100,104,108,112],
        "new_nor48_source_identities":0,
        "repeat_count":REPEATS,
        "armed_auditors":list(ARMS),
        "same_run_five_repeats_per_source":True,
        "strict_authorized_program_only_routing":True,
        "explicit_reentrant_GMP_convolution_injection":True,
        "global_verifier_monkeypatch_used":False,
        "full_exact_root_included_all_arms":True,
        "shared_signatures_session_transport_excluded_all_arms":True,
        "cpu_model":cpu_model(),"python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "gmp_dependency":gmp.backend_metadata(),
        "by_R":groups,
        "overall":overall,
        "original_engineering_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":results,
        "scope":"Historical CPU engineering, no fresh prospective model evaluation. "
                "GMP trace is nonsuccinct. Pinned authority and honest program "
                "and reference provider remain external trust premises."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repeats",type=int,default=REPEATS)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=run(args.repeats)
    raw=canonical(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
    print("NOR48 ORIGINAL CPU BYTES SHA",len(raw),sha256(raw).hexdigest())
    print("NOR48 SUMMARY",result["overall"],result["by_R"])
    print("NOR48 ENGINEERING PASS",result["first_original_engineering_pass"])


if __name__=="__main__":
    main()
