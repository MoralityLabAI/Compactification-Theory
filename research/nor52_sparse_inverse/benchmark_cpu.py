"""NOR52 first 60-source support-compressed exact reciprocal CPU benchmark.

Runs only after an independent public-only SHA source fixture has been frozen.
Four full vector/full NOR45 Merkle-root arms with five interleaved repeats.
A source error stays in the first raw CPU result and fails registered gates.
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
from research.nor51_exact_r_crossover import algebra as prior
from research.nor52_sparse_inverse import algebra,cohort,phase_kernel

REGISTERED="9526627e21b429e7fea1928b3ddfb0e1dadc5e74"
SCHEMA="nor52.first_60_exact_sparse_support_compression_cpu.v1"
FIRST_REPEATS=5
ARMS=("native_full_R_root","native_compressed_R_root",
      "gmp_full_verify_root","gmp_compressed_verify_root")

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def run_arm(arm,source,reference,stride):
    if arm==ARMS[0]:
        return algebra.native_full_root(source)
    if arm==ARMS[1]:
        return algebra.native_compressed_root(source,stride)
    if arm==ARMS[2]:
        return algebra.gmp_full_root(source,reference)
    if arm==ARMS[3]:
        return algebra.gmp_compressed_root(source,reference,stride)
    raise ValueError("outside frozen NOR52 four-arm set")

def median(seq):
    return statistics.median(seq) if seq else None

def summarize(rows):
    if not rows:
        return {"count":0}
    return {
        "count":len(rows),
        "median_native_compressed_over_full":median(
            r["native_compressed_over_full"] for r in rows),
        "median_gmp_compressed_over_full":median(
            r["gmp_compressed_over_full"] for r in rows),
        "median_gmp_full_over_native_full":median(
            r["gmp_full_over_native_full"] for r in rows),
        "median_gmp_compressed_over_native_compressed":median(
            r["gmp_compressed_over_native_compressed"] for r in rows),
        "median_source_arm_ms":{a:median(r["arm_medians_ms"][a] for r in rows) for a in ARMS},
        "median_full_merkle_bytes":median(r["full_inverse_canonical_json_bytes"] for r in rows)
    }

def bootstrap(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["replicate"],r["profile"])].append(r)
    if len(groups)!=10 or any(len(v)!=6 for v in groups.values()):
        return {"status":"incomplete 10-block grid"}
    keys=sorted(groups)
    totals={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    comparison={}
    for i,arm in enumerate(ARMS[1:]):
        rng=random.Random(20261152000+i)
        ratios=[]
        for t in range(4096):
            sample=[x for b in (rng.choice(keys) for _ in keys) for x in groups[b]]
            ratios.append(sum(r["arm_medians_ms"][arm] for r in sample)/
                          sum(r["arm_medians_ms"][ARMS[0]] for r in sample))
        ratios.sort()
        comparison[arm]={
            "observed_secondary_matched_workload_ratio":totals[arm]/totals[ARMS[0]],
            "descriptive_ten_block_95pct_interval":[ratios[102],ratios[3993]]
        }
    return {"source_median_total_ms":totals,
            "resamples":4096,"block_count":10,"secondary_ratios":comparison}

def run(public_file:Path,repeats=FIRST_REPEATS):
    if repeats!=FIRST_REPEATS:
        raise ValueError("first NOR52 requires exactly five interleaved repeats")
    fixture,source_sha=cohort.read_fixture(public_file)
    assert fixture["first_registration_sha"]==REGISTERED
    assert fixture["source_count"]==60
    source_ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(source_ids)==60 and source_ids.isdisjoint(cohort.first_prior_source_ids())
    global_before=nor46.exact_convolution_prefix
    rows=[]
    for ordinal,case in enumerate(fixture["cases"]):
        metadata={k:case[k] for k in ("source_sha256","replicate","length",
                                      "profile","family","stride",
                                      "coefficient_magnitude_bits")}
        try:
            f=algebra.checked_source(
                tuple(int(x) for x in case["source_coefficients_decimal"]),
                case["stride"])
            assert len(f)==case["length"]
            begin=time.perf_counter_ns()
            reference=prior.native_single_inverse(f)
            reference_ns=time.perf_counter_ns()-begin
            if any(type(v) is not int or abs(v).bit_length()>nor46.MAX_COEFFICIENT_BITS
                   for v in reference):
                raise ValueError("first original inverse over coefficient cap")
            bounds=algebra.original_and_compressed_resource_sizes(
                f,reference,case["stride"])
            reference_root=prior.root(reference)
            begin=time.perf_counter_ns()
            witness_json=json.dumps([str(v) for v in reference],
                                    separators=(",",":"),ensure_ascii=True).encode("ascii")
            witness_serial_ns=time.perf_counter_ns()-begin
            # Pre-time exact full vector and complete Merkle-root parity.
            for arm in ARMS:
                actual,root=run_arm(arm,f,reference,case["stride"])
                if actual!=reference or root!=reference_root:
                    raise ValueError(f"first NOR52 pre-time parity failure on {arm}")
            samples={arm:[] for arm in ARMS}
            for repetition in range(FIRST_REPEATS):
                order=list(ARMS)
                seed_text=f"NOR52/INTERLEAVED/v1|{case['source_sha256']}|{repetition}"
                seed=int.from_bytes(sha256(seed_text.encode("ascii")).digest(),"big")
                random.Random(seed).shuffle(order)
                for arm in order:
                    started=time.perf_counter_ns()
                    actual,root=run_arm(arm,f,reference,case["stride"])
                    elapsed_ms=(time.perf_counter_ns()-started)/1e6
                    if actual!=reference or root!=reference_root:
                        raise ValueError(f"timed exact output/root parity failure: {arm}")
                    if nor46.exact_convolution_prefix is not global_before:
                        raise RuntimeError("NOR46 process-global convolution changed")
                    samples[arm].append(elapsed_ms)
            med={arm:median(samples[arm]) for arm in ARMS}
            # The independent NOR47 microphase profiler runs AFTER all
            # registered timed arms. It is a separate diagnostic, not a route.
            phases=phase_kernel.instrumented_exact_product(f,reference)
            if phases["full_packed_bits"]!=bounds["original_selected_packed_bits"]:
                raise ValueError("phase ledger original signed resource bound changed")
            row={
                **metadata,
                "source_success":True,
                "full_exact_inverse_vector_and_merkle_root_equal":True,
                "reference_merkle_root_sha256":reference_root,
                "full_inverse_canonical_json_bytes":len(witness_json),
                "reference_inverse_generation_ns_outside_arms":reference_ns,
                "witness_serialization_ns_outside_arms":witness_serial_ns,
                "original_and_compressed_resource_bounds":bounds,
                "untimed_original_full_gmp_phases":phases,
                "arm_medians_ms":med,
                "five_interleaved_samples_ms":samples,
                "native_compressed_over_full":med[ARMS[1]]/med[ARMS[0]],
                "gmp_compressed_over_full":med[ARMS[3]]/med[ARMS[2]],
                "gmp_full_over_native_full":med[ARMS[2]]/med[ARMS[0]],
                "gmp_compressed_over_native_compressed":med[ARMS[3]]/med[ARMS[1]]
            }
            rows.append(row)
            print("NOR52 FIRST CPU",ordinal+1,"/60",case["profile"],case["length"],
                  "cached native",round(row["native_compressed_over_full"],4),
                  "compressed GMP",round(row["gmp_compressed_over_full"],4),flush=True)
        except Exception as exc:
            rows.append({**metadata,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR52 FIRST CPU SOURCE FAIL",ordinal+1,"/60",
                  type(exc).__name__,str(exc)[:180],flush=True)
    successes=[r for r in rows if r["source_success"]]
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={m:20 for m in cohort.LENGTHS}
    sparse_large=[r for r in successes if r["length"]==3073 and r["stride"]>1]
    dense=[r for r in successes if r["stride"]==1]
    s_sparse=summarize(sparse_large)
    s_dense=summarize(dense)
    all_exact=(len(successes)==60 and all(
        r["full_exact_inverse_vector_and_merkle_root_equal"] for r in successes))
    all_phases=(len(successes)==60 and all(
        r["untimed_original_full_gmp_phases"]["exact_signed_identity"] and
        r["untimed_original_full_gmp_phases"]["reference_kernel_equal"] and
        r["untimed_original_full_gmp_phases"]["phase_total_ns"]==
        sum(r["untimed_original_full_gmp_phases"][k] for k in phase_kernel.PHASES)
        for r in successes))
    gates={
        "sparse_m3073_native_compressed_over_full_under_0p50":(
            len(sparse_large)==16 and s_sparse["median_native_compressed_over_full"]<0.50),
        "sparse_m3073_gmp_compressed_over_full_under_0p50":(
            len(sparse_large)==16 and s_sparse["median_gmp_compressed_over_full"]<0.50),
        "dense_s1_both_compressed_over_full_under_1p10":(
            len(dense)==12 and s_dense["median_native_compressed_over_full"]<1.10
            and s_dense["median_gmp_compressed_over_full"]<1.10),
        "all_exact_output_roots_and_source_free_adversarial_controls":all_exact,
        "frozen_60_source_five_interleaved_reentrant_caps_and_phase_parity":(
            len(successes)==60 and all_phases and
            source_ids.isdisjoint(cohort.first_prior_source_ids()) and
            nor46.exact_convolution_prefix is global_before and
            all(all(len(r["five_interleaved_samples_ms"][arm])==5 for arm in ARMS)
                for r in successes) and all(
                r["original_and_compressed_resource_bounds"]["original_selected_packed_bits"]
                    <=nor46.MAX_PACKED_BITS and
                r["original_and_compressed_resource_bounds"]["compressed_selected_packed_bits"]
                    <=nor46.MAX_PACKED_BITS and
                r["original_and_compressed_resource_bounds"]["original_witness_max_bits"]
                    <=nor46.MAX_COEFFICIENT_BITS for r in successes))
    }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        rss=None
    return {
        "schema":SCHEMA,
        "design_issue":259,
        "registration_first_sha":REGISTERED,
        "first_public_only_source_sha256":source_sha,
        "source_count":60,"source_success_count":len(successes),
        "source_failure_count":60-len(successes),
        "historical_nor51_disjoint":source_ids.isdisjoint(cohort.first_prior_source_ids()),
        "repeats":FIRST_REPEATS,
        "arms":list(ARMS),
        "full_merkle_root_in_all_arm_timings":True,
        "reference_production_serialization_outside_timings":True,
        "full_witness_original_resource_caps_enforced":True,
        "public_only_sha_frozen_upstream_of_timing":True,
        "no_NOR46_global_override":nor46.exact_convolution_prefix is global_before,
        "gmp_dependency":gmp.backend_metadata(),
        "cpu_model":platform.processor() or "unknown",
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "maxrss_kib":rss,
        "overall":summarize(successes),
        "sparse_longest_m3073":s_sparse,
        "dense_stride1_overhead":s_dense,
        "by_length":{str(n):summarize([r for r in successes if r["length"]==n])
                     for n in cohort.LENGTHS},
        "by_profile":{name:summarize([r for r in successes if r["profile"]==name])
                      for name in cohort.PROFILES},
        "by_family":{name:summarize([r for r in successes if r["family"]==name])
                     for name in cohort.FAMILIES},
        "secondary_source_median_workload":bootstrap(successes),
        "first_original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "limitations":"Synthetic public formal-series algebra, transparent exact witnesses, no m3073 NOR45 signed grammar or external semantic provider."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=FIRST_REPEATS)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    x=run(a.public_file,a.repeats)
    raw=canonical(x)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR52 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR52 FIRST ORIGINAL FIVE GATES",x["first_original_five_registered_gates"])
    print("NOR52 FIRST ORIGINAL ENGINEERING PASS",x["first_original_engineering_pass"])

if __name__=="__main__":main()
