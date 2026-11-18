"""NOR51 source-frozen, five-interleaved repeat exact reciprocal CPU frontier.

The first public-only source fixture must arrive from a separate upstream
source job. Benchmark's original scoring is frozen in REGISTRATION.md.
A failed original source becomes a visible error row and FAIL verdict; no
source is silently replaced or retuned.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor51_exact_r_crossover import algebra,cohort

REPEATS=5
ARMS=("native_single_R_root","gmp_verify_reference_root",
      "newton_produce_verify_root")
REGISTERED="83fabe1c34c2bedbba35faa928349f41b964115a"
SCHEMA="nor51.first_source_frozen_48_exact_R_crossover_cpu.v1"
ISSUE=257

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def arm(name,source,witness):
    if name==ARMS[0]:return algebra.direct_with_root(source)
    if name==ARMS[1]:return algebra.gmp_witness_with_root(source,witness)
    if name==ARMS[2]:return algebra.newton_verified_with_root(source)
    raise ValueError("outside registered NOR51 benchmark arm")

def arm_seed(sid,repetition):
    digest=sha256(f"NOR51/ARM-ORDER/v1|{sid}|{repetition}".encode("ascii")).digest()
    return int.from_bytes(digest,"big")

def median(values):
    return statistics.median(values) if values else None

def summarise(rows):
    if not rows:return {"count":0}
    return {
        "count":len(rows),
        "median_gmp_verify_over_direct":median(
            r["gmp_verify_over_direct"] for r in rows),
        "median_newton_end_to_end_over_direct":median(
            r["newton_end_to_end_over_direct"] for r in rows),
        "source_median_ms":{a:median(r["arm_medians_ms"][a] for r in rows) for a in ARMS},
        "median_source_reference_producer_ms":median(
            r["outside_arm_native_reference_generation_ms"] for r in rows),
        "median_source_witness_json_bytes":median(
            r["outside_arm_witness_json_bytes"] for r in rows)
    }

def grouped_bootstrap(rows):
    """Secondary block bootstrap: eight distinct (rep,profile) input blocks."""
    groups=defaultdict(list)
    for r in rows:groups[(r["replicate"],r["profile"])].append(r)
    if len(groups)!=8 or any(len(v)!=6 for v in groups.values()):
        return {"status":"incomplete 8-block grid"}
    keys=sorted(groups)
    observed_totals={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    result={}
    for a in ARMS[1:]:
        rng=random.Random(2026115100+(0 if a==ARMS[1] else 1))
        ratios=[]
        for _ in range(4096):
            sample=[r for _block in keys for r in groups[rng.choice(keys)]]
            ratios.append(sum(r["arm_medians_ms"][a] for r in sample)/
                          sum(r["arm_medians_ms"][ARMS[0]] for r in sample))
        ratios.sort()
        result[a]={
            "observed_matched_workload_ratio":
                observed_totals[a]/observed_totals[ARMS[0]],
            "descriptive_95pct_eight_block_bootstrap":[ratios[102],ratios[3993]]
        }
    return {"total_source_median_ms":observed_totals,"n_block_resamples":4096,
            "secondary_paired_workload_ratios":result}

def run(source_file:Path,repeats=REPEATS):
    if repeats!=REPEATS:raise ValueError("first NOR51 requires exactly 5 repeats")
    public,public_sha=cohort.validate_fixture(source_file)
    assert len(public["cases"])==48 and public["registration_first_commit"]==REGISTERED
    global_before=nor46.exact_convolution_prefix
    results=[]
    for ordinal,row in enumerate(public["cases"]):
        sid=row["source_sha256"]
        prefix={k:row[k] for k in ("source_sha256","replicate","length","profile",
                                    "family","stride","coefficient_magnitude_bits")}
        try:
            source=algebra.normalized(tuple(int(x) for x in row["source_coefficients_decimal"]))
            m=len(source)
            assert m==row["length"]
            start=time.perf_counter_ns()
            reference=algebra.native_single_inverse(source)
            reference_ms=(time.perf_counter_ns()-start)/1e6
            ref_root=algebra.root(reference)
            start=time.perf_counter_ns()
            serialized=json.dumps([str(v) for v in reference],ensure_ascii=True,
                                  separators=(",",":")).encode("ascii")
            serialized_ms=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns()
            newton=algebra.newton_inverse(source)
            newton_ms=(time.perf_counter_ns()-start)/1e6
            if newton!=reference:raise algebra.ExactReciprocalError(
                "Newton inverse differs from independent quadratic reference")
            algebra.checked_inverse_witness(source,reference)
            _,offset,width=gmp._parameters(source,reference)
            packed_bits=m*width
            assert packed_bits<=nor46.MAX_PACKED_BITS
            for a in ARMS:
                got,root=arm(a,source,reference)
                if got!=reference or root!=ref_root:
                    raise algebra.ExactReciprocalError("pre-timing output/root mismatch")
            sample={a:[] for a in ARMS}
            for repetition in range(REPEATS):
                order=list(ARMS)
                random.Random(arm_seed(sid,repetition)).shuffle(order)
                for a in order:
                    begin=time.perf_counter_ns()
                    got,root=arm(a,source,reference)
                    ms=(time.perf_counter_ns()-begin)/1e6
                    if got!=reference or root!=ref_root:
                        raise algebra.ExactReciprocalError("timed output/root mismatch")
                    if nor46.exact_convolution_prefix is not global_before:
                        raise RuntimeError("NOR46 process-global convolution mutated")
                    sample[a].append(ms)
            med={a:median(sample[a]) for a in ARMS}
            result={
                **prefix,
                "source_success":True,
                "full_exact_output_and_root_identical":True,
                "exact_inverse_merkle_root_sha256":ref_root,
                "reference_max_abs_bit_length":max(abs(v).bit_length() for v in reference),
                "gmp_selected_digit_width_bits":width,
                "gmp_full_packed_bits":packed_bits,
                "outside_arm_native_reference_generation_ms":reference_ms,
                "outside_arm_newton_reference_generation_ms":newton_ms,
                "outside_arm_witness_json_serialization_ms":serialized_ms,
                "outside_arm_witness_json_bytes":len(serialized),
                "arm_medians_ms":med,
                "five_interleaved_samples_ms":sample,
                "gmp_verify_over_direct":med[ARMS[1]]/med[ARMS[0]],
                "newton_end_to_end_over_direct":med[ARMS[2]]/med[ARMS[0]]
            }
            results.append(result)
            print("NOR51 FIRST CPU",ordinal+1,"/48",row["profile"],m,
                  "GMP/native",round(result["gmp_verify_over_direct"],4),
                  "Newton-e2e/native",round(result["newton_end_to_end_over_direct"],4),
                  flush=True)
        except Exception as exc:
            results.append({**prefix,"source_success":False,
                            "failure_type":type(exc).__name__,
                            "failure_message":str(exc)[:500]})
            print("NOR51 FIRST CPU SOURCE FAIL",ordinal+1,"/48",
                  type(exc).__name__,str(exc)[:180],flush=True)
    success=[r for r in results if r["source_success"]]
    assert len(results)==48 and len({r["source_sha256"] for r in results})==48
    assert Counter(r["length"] for r in results)=={m:16 for m in cohort.LENGTHS}
    large=[r for r in success if r["length"]==3073]
    paired={}
    for r in success:
        key=(r["replicate"],r["profile"],r["family"])
        paired.setdefault(key,{})[r["length"]]=r["gmp_verify_over_direct"]
    slope=[values[3073]/values[769] for values in paired.values()
           if 3073 in values and 769 in values]
    global_unchanged=nor46.exact_convolution_prefix is global_before
    gates={
        "all_48_exact_inverse_merkle_and_negative_controls":(
            len(success)==48 and all(r["full_exact_output_and_root_identical"] for r in success)),
        "longest_m3073_gmp_verify_over_direct_under_0p75":(
            len(large)==16 and median([r["gmp_verify_over_direct"] for r in large])<.75),
        "longest_m3073_newton_e2e_over_direct_under_0p95":(
            len(large)==16 and median([r["newton_end_to_end_over_direct"] for r in large])<.95),
        "matched_long_over_short_gmp_ratio_slope_under_0p80":(
            len(slope)==16 and median(slope)<.80),
        "public_only_source_first_reentrant_48_five_repeat_caps":(
            public["public_only"] and public["no_inverse_root_trace_or_timings"]
            and len(results)==48 and global_unchanged
            and all(r["gmp_full_packed_bits"]<=nor46.MAX_PACKED_BITS and
                    r["reference_max_abs_bit_length"]<=nor46.MAX_COEFFICIENT_BITS
                    and all(len(r["five_interleaved_samples_ms"][a])==5 for a in ARMS)
                    for r in success) and len(success)==48)
    }
    extra={}
    try:
        import resource
        extra["maxrss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        extra["maxrss_kib"]=None
    return {
        "schema":SCHEMA,
        "issue":ISSUE,"registration_first_commit":REGISTERED,
        "first_frozen_public_only_source_sha256":public_sha,
        "source_count":48,"source_success_count":len(success),
        "source_fail_count":48-len(success),"new_public_only_source_identities":48,
        "cohort_replicates":list(cohort.REPS),
        "cohort_lengths":list(cohort.LENGTHS),
        "cohort_profiles":list(cohort.PROFILES),
        "cohort_families":list(cohort.FAMILIES),
        "three_original_timed_arms":list(ARMS),
        "same_run_interleaved_repetitions":REPEATS,
        "source_only_upstream_precedes_timing":True,
        "exact_reference_native_inverse_produced_outside_arms":True,
        "full_merkle_root_built_all_arms":True,
        "witness_generation_charged_in_newton_end_to_end_arm":True,
        "witness_json_serialization_charged_separately":True,
        "no_global_NOR46_verifier_override":global_unchanged,
        "signed_authorization_in_kernel":False,
        "fresh_native_signed_grammar_extension":False,
        "gmp_metadata":algebra.backend_metadata(),
        "cpu_model":platform.processor() or (Path("/proc/cpuinfo").read_text().split("model name\t: ")[1].split("\n")[0]
                                           if Path("/proc/cpuinfo").exists() and "model name\t: " in
                                              Path("/proc/cpuinfo").read_text() else "unknown"),
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "memory":extra,
        "overall":summarise(success),
        "by_length":{str(m):summarise([r for r in success if r["length"]==m]) for m in cohort.LENGTHS},
        "by_profile":{p:summarise([r for r in success if r["profile"]==p]) for p in cohort.PROFILES},
        "by_family":{f:summarise([r for r in success if r["family"]==f]) for f in cohort.FAMILIES},
        "primary_longest_length":summarise(large),
        "primary_matched_length_ratio_slope":median(slope),
        "primary_paired_cell_count":len(slope),
        "secondary_workload_totals_and_block_uncertainty":grouped_bootstrap(success),
        "original_five_registered_gates":gates,
        "first_engineering_pass":all(gates.values()),
        "source_records":results,
        "limitations":"Exact synthetic single-R arithmetic kernel beyond signed NOR45 grammar at m3073; no real program, provider or session proof."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-file",required=True,type=Path)
    p.add_argument("--repeats",default=REPEATS,type=int)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    x=run(args.source_file,args.repeats)
    raw=canonical(x)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(raw)
    print("NOR51 FIRST ORIGINAL CPU BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR51 ORIGINAL FIRST ENGINEERING GATES",x["original_five_registered_gates"])
    print("NOR51 ORIGINAL FIRST ENGINEERING PASS",x["first_engineering_pass"])

if __name__=="__main__":main()
