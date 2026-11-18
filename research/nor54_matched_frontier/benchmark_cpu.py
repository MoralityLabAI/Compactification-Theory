"""NOR54: first registered same-run compressed native/fast GMP frontier.

Requires an independently SHA-frozen 60-source public-only input artifact
uploaded before timing. Every one of five arms computes a complete full-length
signed integer witness and an independently recomputed full Merkle root.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as original_gmp
from research.nor51_exact_r_crossover import algebra as nor51
from research.nor52_sparse_inverse import algebra as nor52
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import cohort,producer

REGISTERED=cohort.REGISTERED
SCHEMA="nor54.first_60_matched_native_fast_gmp_e2e_cpu.v1"
REPEATS=5
ARMS=(
    "full_native_R_root",
    "compressed_native_R_root",
    "fast_full_gmp_witness_root",
    "fast_compressed_gmp_witness_root",
    "compressed_produce_serialize_fast_verify_root"
)

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def median(vals):
    return statistics.median(vals) if vals else None

def run_arm(name,source,reference,stride):
    if name==ARMS[0]:
        return nor51.direct_with_root(source)
    if name==ARMS[1]:
        return nor52.native_compressed_root(source,stride)
    if name==ARMS[2]:
        return fast_kernel.full_verified_root(source,reference)
    if name==ARMS[3]:
        return fast_kernel.compressed_verified_root(source,reference,stride)
    if name==ARMS[4]:
        return producer.compressed_produce_serialize_fast_verify_root(source,stride)
    raise ValueError("unknown frozen NOR54 arm")

def summarize(rows):
    if not rows:return {"count":0}
    keys=(
        "compressed_native_over_full_native",
        "fast_compressed_verify_over_compressed_native",
        "fast_full_verify_over_full_native",
        "fast_compressed_over_fast_full",
        "produce_serialize_fast_verify_over_compressed_native",
        "end_to_end_over_fast_compressed_verify"
    )
    return {
        "count":len(rows),
        **{"median_"+k:median(r[k] for r in rows) for k in keys},
        "median_arm_ms":{arm:median(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS},
        "median_reference_producer_ns":median(
            r["reference_full_native_generation_ns_outside_arms"] for r in rows),
        "median_root_only_ns":median(
            r["reference_full_merkle_root_ns_outside_arms"] for r in rows),
        "median_full_witness_json_bytes":median(
            r["independent_full_witness_json_bytes"] for r in rows)
    }

def paired_block_bootstrap(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["replicate"],r["profile"])].append(r)
    if len(groups)!=10 or any(len(v)!=6 for v in groups.values()):
        return {"status":"missing registered ten-block source group"}
    keys=sorted(groups)
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    comparisons={}
    for index,arm in enumerate(ARMS):
        if arm==ARMS[1]:continue
        rng=random.Random(202611540000+index)
        ratios=[]
        for _ in range(4096):
            sample=[r for block in (rng.choice(keys) for _ in keys)
                    for r in groups[block]]
            ratios.append(sum(r["arm_medians_ms"][arm] for r in sample)/
                          sum(r["arm_medians_ms"][ARMS[1]] for r in sample))
        ratios.sort()
        comparisons[arm]={
            "whole_workload_source_median_ratio_to_compressed_native":
                totals[arm]/totals[ARMS[1]],
            "descriptive_ten_block_95pct_interval":[ratios[102],ratios[3993]]
        }
    return {
        "source_median_total_ms":totals,
        "bootstrap_repetitions":4096,
        "source_unit_blocks":10,
        "secondary_ratios_to_compressed_native":comparisons
    }

def run(public_file:Path,repeats:int=REPEATS):
    if repeats!=REPEATS:
        raise ValueError("NOR54 original first protocol requires exactly five repeats")
    fixture,fixture_sha=cohort.read_fixture(public_file)
    source_ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(source_ids)==60 and source_ids.isdisjoint(cohort.first_previous_ids())
    original_nor46=nor46.exact_convolution_prefix
    rows=[]
    for ordinal,case in enumerate(fixture["cases"]):
        identity={k:case[k] for k in (
            "source_sha256","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits")}
        try:
            f=nor51.normalized(tuple(int(x) for x in
                                    case["source_coefficients_decimal"]))
            if len(f)!=case["length"]:
                raise ValueError("first original source length mismatch")
            start=time.perf_counter_ns()
            reference=nor51.native_single_inverse(f)
            generation_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            json_wire=producer.canonical_witness_bytes(reference)
            serialization_ns=time.perf_counter_ns()-start
            assert producer.parse_witness_bytes(json_wire,len(f))==reference
            start=time.perf_counter_ns()
            root=native.independently_computed_root(reference)
            root_ns=time.perf_counter_ns()-start
            resource=nor52.original_and_compressed_resource_sizes(
                f,reference,case["stride"])
            assert resource["original_coefficient_count"]==case["length"]
            for name in ARMS:
                actual,actual_root=run_arm(name,f,reference,case["stride"])
                if actual!=reference or actual_root!=root:
                    raise ValueError("first NOR54 pre-timing complete vector/root parity: "+name)
            timings={arm:[] for arm in ARMS}
            for repeat in range(REPEATS):
                names=list(ARMS)
                salt=f"NOR54/ARM-ORDER/v1|{case['source_sha256']}|{repeat}"
                shuffle_seed=int.from_bytes(sha256(salt.encode("ascii")).digest(),"big")
                random.Random(shuffle_seed).shuffle(names)
                for name in names:
                    start=time.perf_counter_ns()
                    actual,actual_root=run_arm(name,f,reference,case["stride"])
                    duration_ms=(time.perf_counter_ns()-start)/1e6
                    if actual!=reference or actual_root!=root:
                        raise ValueError("timed NOR54 full exact vector/root parity "+name)
                    if nor46.exact_convolution_prefix is not original_nor46:
                        raise RuntimeError("NOR46 process-global verifier changed")
                    timings[name].append(duration_ms)
            medians={name:median(timings[name]) for name in ARMS}
            ratios={
                "compressed_native_over_full_native":medians[ARMS[1]]/medians[ARMS[0]],
                "fast_compressed_verify_over_compressed_native":
                    medians[ARMS[3]]/medians[ARMS[1]],
                "fast_full_verify_over_full_native":medians[ARMS[2]]/medians[ARMS[0]],
                "fast_compressed_over_fast_full":medians[ARMS[3]]/medians[ARMS[2]],
                "produce_serialize_fast_verify_over_compressed_native":
                    medians[ARMS[4]]/medians[ARMS[1]],
                "end_to_end_over_fast_compressed_verify":medians[ARMS[4]]/medians[ARMS[3]]
            }
            rows.append({
                **identity,"source_success":True,
                "all_five_full_exact_vectors_and_merkle_roots_equal":True,
                "reference_full_merkle_root_sha256":root,
                "reference_full_native_generation_ns_outside_arms":generation_ns,
                "reference_full_merkle_root_ns_outside_arms":root_ns,
                "independent_full_witness_json_bytes":len(json_wire),
                "independent_full_witness_json_serialization_ns_outside_arms":serialization_ns,
                "full_and_compressed_signed_resource_bounds":resource,
                "arm_medians_ms":medians,
                "five_interleaved_samples_ms":timings,
                **ratios
            })
            print("NOR54 FIRST CPU",ordinal+1,"/60",case["profile"],case["length"],
                  "GMP/native",round(ratios["fast_compressed_verify_over_compressed_native"],4),
                  "end-to-end/native",
                  round(ratios["produce_serialize_fast_verify_over_compressed_native"],4),
                  flush=True)
        except Exception as exc:
            rows.append({**identity,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR54 FIRST SOURCE FAIL",ordinal+1,"/60",
                  type(exc).__name__,str(exc)[:180],flush=True)
    successful=[r for r in rows if r["source_success"]]
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={m:20 for m in cohort.LENGTHS}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    sparse_m3073=[r for r in successful if r["length"]==3073 and r["stride"]>1]
    dense=[r for r in successful if r["stride"]==1]
    sparse_summary=summarize(sparse_m3073)
    dense_summary=summarize(dense)
    unchanged=nor46.exact_convolution_prefix is original_nor46
    gates={
        "sparse_m3073_fast_compressed_gmp_over_compressed_native_under_0p85":(
            len(sparse_m3073)==16 and
            sparse_summary["median_fast_compressed_verify_over_compressed_native"]<.85),
        "sparse_m3073_local_produce_serialize_verify_over_native_under_1p50":(
            len(sparse_m3073)==16 and
            sparse_summary["median_produce_serialize_fast_verify_over_compressed_native"]<1.50),
        "dense_s1_native_and_gmp_compressed_over_full_under_1p10":(
            len(dense)==12 and
            dense_summary["median_compressed_native_over_full_native"]<1.10 and
            dense_summary["median_fast_compressed_over_fast_full"]<1.10),
        "all_60_exact_full_vectors_merkle_and_hostile_source_controls":(
            len(successful)==60 and
            all(r["all_five_full_exact_vectors_and_merkle_roots_equal"] for r in successful)),
        "first_60_source_SHA_five_repeats_caps_and_reentrant_archive":(
            len(successful)==60 and fixture["public_only"] and
            source_ids.isdisjoint(cohort.first_previous_ids()) and unchanged and
            all(all(len(r["five_interleaved_samples_ms"][a])==5 for a in ARMS)
                for r in successful) and
            all(r["full_and_compressed_signed_resource_bounds"][
                "original_selected_packed_bits"]<=nor46.MAX_PACKED_BITS and
                r["full_and_compressed_signed_resource_bounds"][
                    "compressed_selected_packed_bits"]<=nor46.MAX_PACKED_BITS and
                r["full_and_compressed_signed_resource_bounds"][
                    "original_witness_max_bits"]<=nor46.MAX_COEFFICIENT_BITS
                for r in successful))
    }
    try:
        import resource
        maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        maxrss=None
    return {
        "schema":SCHEMA,"design_issue":263,
        "first_registration_sha":REGISTERED,
        "first_public_only_source_sha256":fixture_sha,
        "source_count":60,"source_success_count":len(successful),
        "source_failure_count":60-len(successful),
        "previous_excluded_source_count":168,
        "repeats":REPEATS,
        "five_original_timed_arms":list(ARMS),
        "public_only_sha_frozen_upstream_of_CPU":True,
        "all_arms_have_complete_exact_output_and_original_full_root":True,
        "local_e2e_charges_production_serialization_parse_validation":True,
        "common_session_signature_network_costs_excluded":True,
        "original_full_caps_enforced":True,
        "nor46_process_global_unchanged":unchanged,
        "cpu_model":platform.processor() or "unknown",
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "maxrss_kib":maxrss,
        "gmp_backend":fast_kernel.backend_metadata(),
        "overall":summarize(successful),
        "primary_sixteen_sparse_m3073":sparse_summary,
        "primary_twelve_dense_controls":dense_summary,
        "by_length":{str(m):summarize([r for r in successful if r["length"]==m])
                     for m in cohort.LENGTHS},
        "by_profile":{p:summarize([r for r in successful if r["profile"]==p])
                      for p in cohort.PROFILES},
        "by_family":{f:summarize([r for r in successful if r["family"]==f])
                     for f in cohort.FAMILIES},
        "secondary_ten_block_source_median_workload":paired_block_bootstrap(successful),
        "first_original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "scope":"Synthetic transparent exact algebra; full witness/root verification; m3073 outside signed NOR45 grammar; no remote transport or honest provider proof."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=REPEATS)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    output=run(a.public_file,a.repeats)
    raw=canonical(output)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR54 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR54 FIRST ORIGINAL FIVE GATES",output["first_original_five_registered_gates"])
    print("NOR54 FIRST ENGINEERING PASS",output["first_original_engineering_pass"])

if __name__=="__main__":
    main()
