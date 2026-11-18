"""NOR53: first 60-source exact GMP pack/unpack kernel CPU comparison.

Run only with a SHA-frozen upstream public-only source artifact. All four
five-repeat arms validate full exact integer outputs and build full Merkle roots.
The original five engineering gates and no-retune protocol are preregistered.
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
from research.nor51_exact_r_crossover import algebra as native_reference
from research.nor52_sparse_inverse import algebra as old_sparse
from research.nor53_native_pack import fast_kernel,cohort

REGISTERED=cohort.REGISTERED
SCHEMA="nor53.first_60_source_native_gmp_packed_cpu.v1"
REPEATS=5
ARMS=("original_full_gmp_root","fast_full_gmp_root",
      "original_compressed_gmp_root","fast_compressed_gmp_root")

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def med(values):
    return statistics.median(values) if values else None

def arm(name,source,witness,stride):
    if name==ARMS[0]:
        return old_sparse.gmp_full_root(source,witness)
    if name==ARMS[1]:
        return fast_kernel.full_verified_root(source,witness)
    if name==ARMS[2]:
        return old_sparse.gmp_compressed_root(source,witness,stride)
    if name==ARMS[3]:
        return fast_kernel.compressed_verified_root(source,witness,stride)
    raise ValueError("out-of-registered-arm operation")

def summarize(rows):
    if not rows:return {"count":0}
    return {
        "count":len(rows),
        "median_fast_full_over_original_full":med(
            r["fast_full_over_original_full"] for r in rows),
        "median_fast_compressed_over_original_compressed":med(
            r["fast_compressed_over_original_compressed"] for r in rows),
        "median_original_compressed_over_original_full":med(
            r["original_compressed_over_original_full"] for r in rows),
        "median_fast_compressed_over_original_full":med(
            r["fast_compressed_over_original_full"] for r in rows),
        "median_ms_by_arm":{a:med(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    }

def workload_bootstrap(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["replicate"],r["profile"])].append(r)
    if len(groups)!=10 or any(len(v)!=6 for v in groups.values()):
        return {"status":"incomplete fixed ten-block factorial cohort"}
    keys=sorted(groups)
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    comparisons={}
    for j,a in enumerate(ARMS[1:]):
        rng=random.Random(202611530053+j)
        ratios=[]
        for _ in range(4096):
            sample=[r for block in (rng.choice(keys) for _ in keys)
                    for r in groups[block]]
            ratios.append(sum(r["arm_medians_ms"][a] for r in sample)/
                          sum(r["arm_medians_ms"][ARMS[0]] for r in sample))
        ratios.sort()
        comparisons[a]={
            "whole_workload_source_median_ratio":totals[a]/totals[ARMS[0]],
            "descriptive_ten_block_95pct_interval":[ratios[102],ratios[3993]]
        }
    return {
        "source_median_total_ms":totals,
        "bootstrap_repetitions":4096,
        "source_unit_blocks":10,
        "secondary_ratios_vs_original_full":comparisons
    }

def run(public_fixture:Path,repeats:int=REPEATS):
    if repeats!=REPEATS:raise ValueError("NOR53 first run exactly five repeats")
    fixture,source_sha=cohort.read_fixture(public_fixture)
    source_ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(source_ids)==60
    assert source_ids.isdisjoint(cohort.original_source_ids())
    original_function=nor46.exact_convolution_prefix
    rows=[]
    for ordinal,case in enumerate(fixture["cases"]):
        label={k:case[k] for k in ("source_sha256","replicate","length",
                                   "profile","family","stride",
                                   "coefficient_magnitude_bits")}
        try:
            f=native_reference.normalized(tuple(int(v) for v in
                           case["source_coefficients_decimal"]))
            assert len(f)==case["length"]
            start=time.perf_counter_ns()
            witness=native_reference.native_single_inverse(f)
            reference_production_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            serialized=json.dumps([str(v) for v in witness],
                                  separators=(",",":"),ensure_ascii=True).encode("ascii")
            serialization_ns=time.perf_counter_ns()-start
            expected_root=native_reference.root(witness)
            bounds=old_sparse.original_and_compressed_resource_sizes(
                f,witness,case["stride"])
            # The independent originally specified signed GMP and new fast
            # kernels must agree exactly before timing the four verifier arms.
            for name in ARMS:
                vals,root=arm(name,f,witness,case["stride"])
                if vals!=witness or root!=expected_root:
                    raise ValueError("pre-timing complete output/Merkle mismatch in "+name)
            sample={a:[] for a in ARMS}
            for rep in range(REPEATS):
                names=list(ARMS)
                order_hash=sha256(
                    f"NOR53/ARM-ORDER/v1|{case['source_sha256']}|{rep}".encode("ascii")).digest()
                random.Random(int.from_bytes(order_hash,"big")).shuffle(names)
                for name in names:
                    began=time.perf_counter_ns()
                    vals,root=arm(name,f,witness,case["stride"])
                    elapsed=(time.perf_counter_ns()-began)/1e6
                    if vals!=witness or root!=expected_root:
                        raise ValueError("timed exact vector/root mismatch "+name)
                    if nor46.exact_convolution_prefix is not original_function:
                        raise RuntimeError("NOR46 global exact convolution modified")
                    sample[name].append(elapsed)
            medians={a:med(sample[a]) for a in ARMS}
            rows.append({
                **label,"source_success":True,
                "all_four_full_integer_vectors_and_merkle_roots_equal":True,
                "original_full_merkle_root_sha256":expected_root,
                "full_witness_canonical_json_bytes":len(serialized),
                "reference_witness_generation_ns_outside_arms":reference_production_ns,
                "reference_witness_serialization_ns_outside_arms":serialization_ns,
                "resource_bounds":bounds,
                "arm_medians_ms":medians,
                "five_interleaved_samples_ms":sample,
                "fast_full_over_original_full":medians[ARMS[1]]/medians[ARMS[0]],
                "fast_compressed_over_original_compressed":medians[ARMS[3]]/medians[ARMS[2]],
                "original_compressed_over_original_full":medians[ARMS[2]]/medians[ARMS[0]],
                "fast_compressed_over_original_full":medians[ARMS[3]]/medians[ARMS[0]]
            })
            print("NOR53 FIRST CPU",ordinal+1,"/60",case["profile"],case["length"],
                  "fast/full",round(rows[-1]["fast_full_over_original_full"],4),
                  "fast compressed/old compressed",
                  round(rows[-1]["fast_compressed_over_original_compressed"],4),
                  flush=True)
        except Exception as exc:
            rows.append({**label,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR53 FIRST SOURCE FAIL",ordinal+1,"/60",
                  type(exc).__name__,str(exc)[:180],flush=True)
    success=[r for r in rows if r["source_success"]]
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={m:20 for m in cohort.LENGTHS}
    high=[r for r in success if r["length"]==3073 and
          r["profile"] in ("s4-b8","s4-b16")]
    sparse=[r for r in success if r["stride"]>1]
    original_unchanged=nor46.exact_convolution_prefix is original_function
    valid=(len(success)==60 and all(r["all_four_full_integer_vectors_and_merkle_roots_equal"]
         for r in success))
    gates={
        "all_60_fast_full_over_original_gmp_under_0p70":(
            len(success)==60 and summarize(success)["median_fast_full_over_original_full"]<.70),
        "longest_m3073_high_bit_fast_full_over_original_under_0p50":(
            len(high)==8 and summarize(high)["median_fast_full_over_original_full"]<.50),
        "all_48_sparse_fast_compressed_over_original_under_0p85":(
            len(sparse)==48 and summarize(sparse)[
                "median_fast_compressed_over_original_compressed"]<.85),
        "all_exact_full_merkle_roots_and_hostile_witness_tests":valid,
        "first_source_60_reentrant_caps_five_samples_original_sha_receipt":(
            len(success)==60 and fixture["public_only"] and
            source_ids.isdisjoint(cohort.original_source_ids()) and original_unchanged
            and all(all(len(r["five_interleaved_samples_ms"][a])==5 for a in ARMS)
                    for r in success)
            and all(0<r["resource_bounds"]["original_selected_packed_bits"]<=
                    nor46.MAX_PACKED_BITS and
                    0<r["resource_bounds"]["compressed_selected_packed_bits"]<=
                    nor46.MAX_PACKED_BITS and
                    r["resource_bounds"]["original_witness_max_bits"]<=
                    nor46.MAX_COEFFICIENT_BITS for r in success))
    }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:rss=None
    return {
        "schema":SCHEMA,"design_issue":261,
        "first_registration_sha":REGISTERED,
        "first_public_only_source_sha256":source_sha,
        "source_count":60,"source_success_count":len(success),
        "source_failure_count":60-len(success),
        "historical_exclusion_count":108,
        "repeats":REPEATS,
        "four_original_timed_arms":list(ARMS),
        "gmp_dependency_and_fast_backend":fast_kernel.backend_metadata(),
        "source_only_sha_frozen_upstream_of_CPU":True,
        "all_arms_have_full_vector_and_original_merkle_root":True,
        "reference_witness_production_serialization_excluded_all_arms":True,
        "original_full_and_compressed_resource_caps_enforced":True,
        "nor46_global_convolution_unchanged":original_unchanged,
        "cpu_model":platform.processor() or "unknown",
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "maxrss_kib":rss,
        "overall":summarize(success),
        "primary_longest_high_width":summarize(high),
        "primary_sparse":summarize(sparse),
        "by_length":{str(m):summarize([r for r in success if r["length"]==m])
                     for m in cohort.LENGTHS},
        "by_profile":{p:summarize([r for r in success if r["profile"]==p])
                      for p in cohort.PROFILES},
        "by_family":{f:summarize([r for r in success if r["family"]==f])
                     for f in cohort.FAMILIES},
        "secondary_ten_block_workload":workload_bootstrap(success),
        "first_original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "scope":"Exact synthetic signed formal-series witness; original full Merkle roots; outside m3073 signed program grammar and deployed proof security."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",required=True,type=Path)
    p.add_argument("--repeats",type=int,default=REPEATS)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    result=run(a.public_file,a.repeats)
    raw=canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR53 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR53 ORIGINAL FIRST FIVE GATES",result["first_original_five_registered_gates"])
    print("NOR53 ORIGINAL FIRST ENGINEERING PASS",result["first_original_engineering_pass"])

if __name__=="__main__":
    main()
