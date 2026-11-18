"""NOR-47 first source-frozen GMP/Python exact trace CPU comparison.

Fixed 16 NOR46 first archived source identities, repeat count 5. All four
arms are timed on the SAME runner, including complete trace/output Merkle
root in every verifier arm. Signature/session transport is excluded from
all timing arms as in the original NOR46 benchmark and verified separately
by the unchanged NOR46 signed-source test suite.

The global function switch is BENCHMARK-ONLY and single-threaded. This is
not a production API or cryptographic proof system.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor45_authorized_program import historical_smoke as nor45
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_producer as producer
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor46_exact_integer_trace import original_archive_audit as archive
from research.nor47_gmp_exact_convolution import gmp_kernel as accelerated

REPEAT=5
ARM_IDS=("nor45_direct","nor46_python_packed",
         "nor47_gmp_packed","nor47_python_pack_gmp_multiply")
CPU_FIRST_SHA="4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64"
CPU_FIRST_BYTES=11884


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def trial_seed(*items):
    return int.from_bytes(sha256("|".join(map(str,items)).encode()).digest(),"big")


@contextmanager
def benchmark_kernel(implementation):
    """Do not call concurrently. Restore verifier code even after exceptions."""
    saved=nor46.exact_convolution_prefix
    if implementation is not None:
        nor46.exact_convolution_prefix=implementation
    try:yield
    finally:nor46.exact_convolution_prefix=saved


def cpu_model():
    try:
        for row in Path("/proc/cpuinfo").read_text().splitlines():
            if row.startswith("model name"):
                return row.split(":",1)[1].strip()
    except OSError:pass
    return platform.processor() or "unknown"


def verify_arm(arm,transcript,query,program,d):
    if arm=="nor45_direct":
        value=native.execute_authorized_program(program,query,d)
    else:
        backend={
            "nor46_python_packed":None,
            "nor47_gmp_packed":accelerated.full_gmp,
            "nor47_python_pack_gmp_multiply":accelerated.hybrid_gmp
        }[arm]
        with benchmark_kernel(backend):
            value=nor46.verified_trace_values(transcript,query,program,d)
    return value,native.independently_computed_root(value)


def run(repeats=REPEAT):
    if repeats!=REPEAT:
        raise ValueError("first original NOR47 benchmark requires five repeats")
    first=archive.audit()
    assert first["original_first_receipt_sha256"]==(
        "d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052")
    cpu_archive=ROOT/"research/nor46_exact_integer_trace/results/nor46_first_historical_cpu_cost.json"
    first_cpu=cpu_archive.read_bytes()
    assert len(first_cpu)==CPU_FIRST_BYTES
    assert sha256(first_cpu).hexdigest()==CPU_FIRST_SHA
    prior_benchmark=json.loads(first_cpu)
    assert prior_benchmark["historical_source_pairs"]==16
    archived={r["source_sha256"]:r for r in prior_benchmark["source_records"]}
    from research.nor46_exact_integer_trace import original_archive_audit
    original_receipt=json.loads(original_archive_audit.RESULT.read_bytes())
    signed_roots={row["source_sha256"]:row["signed_program_root_sha256"]
                  for row in original_receipt["source_records"]}
    assert len(archived)==len(signed_roots)==16
    fixture,graded=nor45.load_original()
    source_pairs=[
        (row,result) for row,result in zip(
            fixture["cases"],graded["source_records"],strict=True)
        if row["full_public_input_sha256"] in archived
    ]
    assert len(source_pairs)==16
    assert Counter(r["pair_replicate"] for r,_ in source_pairs)=={
        100:4,104:4,108:4,112:4}
    records=[]
    for ordinal,(row,record) in enumerate(source_pairs):
        sha=row["full_public_input_sha256"]
        assert record["public_sha256"]==sha
        proposal=nor42.nor37.public_proposal(row)
        clean=proposal["pred"]
        _,hidden,beta,hiddenop=nor42.restore_worlds(row,clean)
        program=nor45.spec_from_public_proposal(
            proposal,row["degree"],beta,hiddenop)
        query=tuple(row["public_query_input"])
        transcript=producer.transcript(program,query,row["degree"])
        assert producer.trace_json_bytes(transcript)==archived[sha]["trace_bytes"]
        assert nor46.verified_trace_values(
            transcript,query,program,row["degree"])==hidden
        expected_root=native.independently_computed_root(hidden)
        assert expected_root==signed_roots[sha]
        # R polynomial traces are exact signed integers under the same
        # independently checked original NOR46 source and program.
        R=tuple(int(s) for s in transcript["unary_transforms"]["R"]) if (
            "R" in transcript["unary_transforms"]) else None
        times={name:[] for name in ARM_IDS}
        kernel_times={name:[] for name in (
            "nor46_python_packed","nor47_gmp_packed",
            "nor47_python_pack_gmp_multiply")} if R is not None else {}
        for repetition in range(repeats):
            order=list(ARM_IDS)
            random.Random(trial_seed("NOR47/ARM-ORDER",sha,repetition)).shuffle(order)
            for arm in order:
                t0=time.perf_counter_ns()
                values,root=verify_arm(
                    arm,transcript,query,program,row["degree"])
                ms=(time.perf_counter_ns()-t0)/1e6
                assert values==hidden and root==expected_root
                times[arm].append(ms)
            if R is not None:
                korder=list(kernel_times)
                random.Random(trial_seed("NOR47/KERNEL-ORDER",sha,repetition)).shuffle(korder)
                for arm in korder:
                    fn={
                        "nor46_python_packed":nor46.exact_convolution_prefix,
                        "nor47_gmp_packed":accelerated.full_gmp,
                        "nor47_python_pack_gmp_multiply":accelerated.hybrid_gmp
                    }[arm]
                    t0=time.perf_counter_ns()
                    computed=fn(query,R)
                    ms=(time.perf_counter_ns()-t0)/1e6
                    assert computed[0]==1 and all(v==0 for v in computed[1:])
                    kernel_times[arm].append(ms)
        med={arm:statistics.median(t) for arm,t in times.items()}
        kernel_med={arm:statistics.median(t)
                    for arm,t in kernel_times.items()}
        assert len(med)==4 and all(len(t)==repeats for t in times.values())
        records.append({
            "source_sha256":sha,
            "replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_coefficients":len(hidden),
            "has_reciprocal_R":"R" in transcript["unary_transforms"],
            "full_trace_bytes":producer.trace_json_bytes(transcript),
            "all_signed_exact_integer_and_root_parity":True,
            "nor45_original_median_ms":med["nor45_direct"],
            "nor46_python_median_ms":med["nor46_python_packed"],
            "nor47_gmp_median_ms":med["nor47_gmp_packed"],
            "nor47_hybrid_median_ms":med["nor47_python_pack_gmp_multiply"],
            "ratios":{
                "gmp_over_nor45":med["nor47_gmp_packed"]/med["nor45_direct"],
                "gmp_over_nor46":med["nor47_gmp_packed"]/med["nor46_python_packed"],
                "hybrid_over_nor45":med["nor47_python_pack_gmp_multiply"]/med["nor45_direct"],
                "hybrid_over_nor46":med["nor47_python_pack_gmp_multiply"]/med["nor46_python_packed"]
            },
            "kernel_R_only_median_ms":kernel_med,
            "randomized_interleaved_arm_samples_ms":times,
            "randomized_interleaved_R_kernel_samples_ms":kernel_times
        })
        print("NOR47 historical source",ordinal+1,"/16 D",row["degree"],
              "R",records[-1]["has_reciprocal_R"],
              "GMP over NOR45",round(records[-1]["ratios"]["gmp_over_nor45"],4),
              flush=True)
    def summary(items):
        return {
            "n":len(items),
            "median_nor45_direct_ms":statistics.median(
                r["nor45_original_median_ms"] for r in items),
            "median_nor46_python_ms":statistics.median(
                r["nor46_python_median_ms"] for r in items),
            "median_nor47_gmp_ms":statistics.median(
                r["nor47_gmp_median_ms"] for r in items),
            "median_nor47_hybrid_ms":statistics.median(
                r["nor47_hybrid_median_ms"] for r in items),
            "median_per_source_gmp_over_nor45":statistics.median(
                r["ratios"]["gmp_over_nor45"] for r in items),
            "median_per_source_gmp_over_nor46":statistics.median(
                r["ratios"]["gmp_over_nor46"] for r in items),
            "median_per_source_hybrid_over_nor45":statistics.median(
                r["ratios"]["hybrid_over_nor45"] for r in items),
            "median_full_trace_bytes":statistics.median(
                r["full_trace_bytes"] for r in items)
        }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        rss=None
    overall=summary(records)
    return {
        "schema":"nor47.first_archived_gmp_exact_convolution_cpu.v1",
        "design_issue":248,
        "nor46_original_first_trace_sha256":first["original_first_receipt_sha256"],
        "nor46_original_first_cpu_sha256":CPU_FIRST_SHA,
        "historical_source_pairs":16,
        "source_selection":[100,104,108,112],
        "new_prospective_nor47_source_ids":0,
        "fixed_repeat_count_per_arm":repeats,
        "timing_method":"interleaved seeded arm permutation per source/repetition; median per arm",
        "exact_integer_arithmetic_without_modular_reduction":True,
        "all_signature_program_source_and_root_semantics_unchanged":True,
        "shared_signature_session_transport_excluded_in_all_timings":True,
        "full_merkle_root_computation_included_in_all_verifier_arms":True,
        "randomized_proof_soundness_claim":False,
        "claimed_succinctness":False,
        "gmp_backend":accelerated.backend_metadata(),
        "cpu_model":cpu_model(),"python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "process_wide_maxrss_kib":rss,
        "all_first_source_parity":True,
        "first_engineering_targets":{
            "gmp_faster_than_nor45":overall["median_per_source_gmp_over_nor45"]<1,
            "gmp_at_least_twofold_faster_than_python_trace":
                overall["median_per_source_gmp_over_nor46"]<.5
        },
        "first_engineering_pass":(
            overall["median_per_source_gmp_over_nor45"]<1 and
            overall["median_per_source_gmp_over_nor46"]<.5),
        "summary":overall,
        "by_degree":{str(d):summary([row for row in records if row["degree"]==d])
                     for d in (512,1024)},
        "by_reciprocal_R":{str(on):summary([
            row for row in records if row["has_reciprocal_R"]==on])
            for on in (True,False)},
        "source_records":records,
        "limits":"Historical engineering benchmark, no new study sources. "
                 "GMP exact integer arithmetic not a succinct cryptographic proof. "
                 "Process-level RSS is cumulative; only test keys supplied."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repeats",type=int,default=REPEAT)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    report=run(args.repeats)
    raw=canonical(report)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
    print("NOR47 FIRST HISTORICAL CPU BYTES SHA",len(raw),sha256(raw).hexdigest())
    print("NOR47 SUMMARY",report["summary"])
    print("NOR47 ENGINEERING PASS",report["first_engineering_pass"])


if __name__=="__main__":
    main()
