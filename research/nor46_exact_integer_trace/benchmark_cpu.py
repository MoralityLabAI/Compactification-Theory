"""NOR-46 exploratory hosted CPU benchmark on 16 fixed archived NOR42 pairs.

No new source identities, no new science gate, no cryptographic proof claim.
Separately measures proof construction, independent full trace arithmetic,
and original NOR45 full execution. Both verifier arms rebuild exactly the
same Merkle root; signature/JSON/network overhead is excluded from this
arithmetic+root comparison and is shared under NOR45 session provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor45_authorized_program import historical_smoke as prior
from research.nor45_authorized_program import independent_execution_verifier as exact
from research.nor46_exact_integer_trace import trace_producer as proof
from research.nor46_exact_integer_trace import trace_verifier as checker
from research.nor46_exact_integer_trace import original_archive_audit as archived


def canonical(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def sample_cpu_model():
    try:
        for row in Path("/proc/cpuinfo").read_text().splitlines():
            if row.startswith("model name"):
                return row.split(":",1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def measure(f,repeats):
    durations=[];answer=None
    for _ in range(repeats):
        start=time.perf_counter_ns()
        candidate=f()
        durations.append((time.perf_counter_ns()-start)/1e6)
        if answer is not None and answer!=candidate:
            raise AssertionError("nondeterministic trace/execution result")
        answer=candidate
    return statistics.median(durations),answer


def run(repeats=3):
    if not 1<=repeats<=10:raise ValueError("invalid bounded benchmark repetitions")
    expected=archived.audit()
    source,grade=prior.load_original()
    authorized={r["source_sha256"]:r for r in json.loads(
        archived.RESULT.read_bytes())["source_records"]}
    selection=[(row,record) for row,record in zip(
        source["cases"],grade["source_records"],strict=True)
        if row["full_public_input_sha256"] in authorized]
    assert len(selection)==16
    rows=[]
    for idx,(row,receipt) in enumerate(selection):
        sha=row["full_public_input_sha256"]
        assert receipt["public_sha256"]==sha
        public=nor42.nor37.public_proposal(row)
        clean=public["pred"]
        _,hidden,beta,op=nor42.restore_worlds(row,clean)
        spec=prior.spec_from_public_proposal(public,row["degree"],beta,op)
        query=tuple(row["public_query_input"])
        constructed_ms,transcript=measure(
            lambda:proof.transcript(spec,query,row["degree"]),repeats)
        trace_ms,trace_result=measure(
            lambda:(lambda values:(
                values,exact.independently_computed_root(values)))(
                checker.verified_trace_values(
                    transcript,query,spec,row["degree"])),repeats)
        original_ms,original_result=measure(
            lambda:(lambda values:(
                values,exact.independently_computed_root(values)))(
                exact.execute_authorized_program(
                    spec,query,row["degree"])),repeats)
        assert trace_result==original_result
        assert trace_result[0]==hidden
        assert transcript["output_coefficients"]==[str(v) for v in hidden]
        assert original_result[1]==authorized[sha][
            "signed_program_root_sha256"]
        assert proof.certificate_sha256(transcript)==authorized[sha][
            "full_trace_sha256"]
        rows.append({
            "source_sha256":sha,
            "replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_coefficients":len(hidden),
            "has_reciprocal_R":authorized[sha]["has_reciprocal_R"],
            "trace_bytes":proof.trace_json_bytes(transcript),
            "prover_trace_construction_median_ms":constructed_ms,
            "original_full_execution_plus_merkle_median_ms":original_ms,
            "new_exact_full_trace_plus_merkle_median_ms":trace_ms,
            "trace_verifier_to_original_runtime_ratio":(
                trace_ms/original_ms if original_ms>0 else None),
            "full_exact_integer_and_root_parity":True
        })
        print("NOR46 benchmark",idx+1,"/16",
              "D",row["degree"],"R",rows[-1]["has_reciprocal_R"],
              "prover",round(constructed_ms,3),
              "old_verify",round(original_ms,3),
              "trace_verify",round(trace_ms,3),flush=True)
    assert len(rows)==16
    def stats(part):
        return {
            "n":len(part),
            "median_prover_ms":statistics.median(
                x["prover_trace_construction_median_ms"] for x in part),
            "median_original_verify_ms":statistics.median(
                x["original_full_execution_plus_merkle_median_ms"] for x in part),
            "median_new_full_trace_verify_ms":statistics.median(
                x["new_exact_full_trace_plus_merkle_median_ms"] for x in part),
            "median_new_over_old_verification_ratio":statistics.median(
                x["trace_verifier_to_original_runtime_ratio"] for x in part),
            "mean_trace_bytes":statistics.mean(x["trace_bytes"] for x in part)
        }
    all_rows=rows
    # Linux ru_maxrss is cumulative process peak, not per-arm peak memory.
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        rss=None
    return {
        "schema":"nor46.exploratory_16_archive_cpu_trace_cost.v1",
        "archived_first_receipt_sha256":expected["original_first_receipt_sha256"],
        "source_selection":sorted(set(r["replicate"] for r in rows)),
        "historical_source_pairs":16,
        "new_nor46_prospective_source_pairs":0,
        "test_key_or_model_honesty_not_established":True,
        "proof_system":"transparent exact integers, full trace, non-succinct",
        "repetitions_per_timed_arm":repeats,
        "timing_method":"time.perf_counter_ns, median, source by source",
        "hardware_cpu":sample_cpu_model(),
        "logical_cpu_count":os.cpu_count(),
        "python_version":sys.version.split()[0],
        "process_cumulative_peak_rss_kib_linux":rss,
        "signature_session_transport_excluded_from_verification_timing":True,
        "root_recomputation_included_in_both_verification_arms":True,
        "summary":stats(all_rows),
        "by_degree":{str(d):stats([r for r in rows if r["degree"]==d])
                     for d in (512,1024)},
        "by_reciprocal_R":{str(kind):stats([
            r for r in rows if r["has_reciprocal_R"]==kind])
            for kind in (True,False)
            if any(r["has_reciprocal_R"]==kind for r in rows)},
        "source_records":rows,
        "limits":"CPU scheduling and allocator state vary between hosted runs; "
                 "no GPU, transport, hidden signing-key or deployed entropy. "
                 "Full trace bytes may exceed source/output payload."
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--repeats",type=int,default=3)
    a.add_argument("--out",type=Path)
    args=a.parse_args()
    report=run(args.repeats)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(report))
    print("NOR46 HISTORICAL CPU COST",report["summary"],flush=True)
    print("NOR46 BY DEGREE",report["by_degree"],flush=True)


if __name__=="__main__":
    main()
