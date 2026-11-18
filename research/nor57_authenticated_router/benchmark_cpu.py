"""NOR57 first registered 60-source same-envelope authenticated CPU selector.

Runs only after a distinct upstream source-only job freezes all new source IDs
and the fixture SHA. Each timed arm performs all signed header/key/nonce/source
checks, complete witness decoding and full original Merkle reconstruction.
Five exact same-wire interleaved repetitions for every source and arm.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as prior
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import signed_v2
from research.nor57_authenticated_router import accept,cohort

SCHEMA="nor57.first_signed_v2_public_source_selector_cpu.v1"
REGISTERED=cohort.REGISTERED
REPEATS=5
ARMS=(
    "always_fast_GMP_signed_v2",
    "always_native_recompute_signed_v2",
    "frozen_public_selector_signed_v2"
)
MODE_BY_ARM={
    ARMS[0]:"always_gmp",
    ARMS[1]:"always_native",
    ARMS[2]:"frozen_selector"
}

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def median(vals):
    return statistics.median(vals) if vals else None

def summarize(rows):
    if not rows:return {"count":0}
    keys=(
        "always_native_over_always_GMP",
        "frozen_selector_over_always_GMP",
        "frozen_selector_over_always_native"
    )
    return {
        "count":len(rows),
        **{"median_sourcewise_"+k:median(r[k] for r in rows) for k in keys},
        "median_ms_by_arm":{arm:median(r["arm_medians_ms"][arm] for r in rows)
                            for arm in ARMS},
        "sum_source_median_ms_by_arm":{
            arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS
        },
        "selector_native_case_count":sum(r["frozen_public_selector_decision"]==accept.NATIVE
                                         for r in rows),
        "selector_gmp_case_count":sum(r["frozen_public_selector_decision"]==accept.GMP
                                      for r in rows)
    }

def workload_bootstrap(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row["replicate"],row["profile"])].append(row)
    if len(groups)!=10 or any(len(v)!=6 for v in groups.values()):
        return {"status":"incomplete prospective ten-block source grouping"}
    keys=sorted(groups)
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    comparisons={}
    for i,arm in enumerate(ARMS[1:],start=1):
        rng=random.Random(202611570000+i)
        boots=[]
        for _ in range(4096):
            sampled=[r for block in (rng.choice(keys) for _ in keys)
                     for r in groups[block]]
            boots.append(sum(r["arm_medians_ms"][arm] for r in sampled)/
                         sum(r["arm_medians_ms"][ARMS[0]] for r in sampled))
        boots.sort()
        comparisons[arm]={
            "ratio_of_sum_of_60_source_medians":totals[arm]/totals[ARMS[0]],
            "descriptive_ten_block_95pct_interval":[boots[102],boots[3993]]
        }
    return {
        "sum_of_source_median_ms":totals,
        "matched_source_block_count":10,"bootstrap_repeats":4096,
        "secondary_descriptive_ratios_to_always_GMP":comparisons
    }

def prebuild_same_wire_batch(reference,source,source_sha,stride,expected_root):
    records=[]
    for rep in range(REPEATS):
        context=signed_v2.make_context_v2(
            source=source,source_sha256=source_sha,
            stride=stride,expected_root_sha256=expected_root,
            session_id="NOR57-matched-signed-"+source_sha[:24],
            sequence=rep+1
        )
        start=time.perf_counter_ns()
        wire=signed_v2.make_signed_envelope_v2(reference,context)
        signer_ns=time.perf_counter_ns()-start
        head,payload,signature,unsigned=v1.inspect_header(wire)
        if head!=signed_v2.expected_header(context,payload) or len(signature)!=64:
            raise ValueError("prebuilt signed v2 witness failed original header parity")
        records.append({
            "wire":wire,"context":context,
            "full_signed_wire_bytes":len(wire),
            "reduced_witness_payload_bytes":len(payload),
            "fixture_signer_encode_ns":signer_ns,
            "verifier_nonce_sha256":sha256(bytes.fromhex(context.nonce)).hexdigest()
        })
    return records

def run_arm(name,record,guard):
    return accept.verify_same_signed_envelope(
        record["wire"],record["context"],guard,MODE_BY_ARM[name])

def run(public_file:Path,repeats:int=REPEATS):
    if repeats!=REPEATS:
        raise ValueError("NOR57 first frozen protocol requires precisely five repeats")
    fixture,public_sha=cohort.read_fixture(public_file)
    ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(ids)==60 and ids.isdisjoint(cohort.historical_ids())
    original_process_convolution=nor46.exact_convolution_prefix
    rows=[]
    for ordinal,case in enumerate(fixture["cases"],start=1):
        label={k:case[k] for k in (
            "source_sha256","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits")}
        try:
            f=prior.normalized(tuple(int(x) for x in
                  case["source_coefficients_decimal"]))
            if len(f)!=case["length"]:
                raise ValueError("original source coefficient count mismatched")
            decision=accept.select_from_public_source(f,case["stride"])
            input_bitwidth=max(abs(x).bit_length() for x in f)
            start=time.perf_counter_ns()
            reference=prior.native_single_inverse(f)
            full_reference_generation_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            full_json=producer.canonical_witness_bytes(reference)
            full_json_ns=time.perf_counter_ns()-start
            expected_root=prior.root(reference)
            bounds=support.original_and_compressed_resource_sizes(
                f,reference,case["stride"])
            records=prebuild_same_wire_batch(
                reference,f,case["source_sha256"],case["stride"],expected_root)
            if len({r["verifier_nonce_sha256"] for r in records})!=REPEATS:
                raise ValueError("repeated signer challenge nonce in source")
            # NOR56 signed-v2 original verifier is an independent reference
            # check on one of the prebuilt signed wires, outside original
            # timing and on a fresh private replay guard.
            original_result=signed_v2.verify_signed_envelope_v2(
                records[0]["wire"],records[0]["context"],v1.ReplayGuard())
            if original_result!=(reference,expected_root):
                raise ValueError("original NOR56 signed verifier disagreed")
            # Every new arm validates the exact same signed wire + trusted
            # context before first timed source/repetition sample.
            pre={arm:v1.ReplayGuard() for arm in ARMS}
            for arm in ARMS:
                if run_arm(arm,records[0],pre[arm])!=(reference,expected_root):
                    raise ValueError("pretime signed full vector/root mismatch "+arm)
            guards={arm:v1.ReplayGuard() for arm in ARMS}
            samples={arm:[] for arm in ARMS}
            for rep in range(REPEATS):
                order=list(ARMS)
                salt=f"NOR57/ARM-ORDER/v1|{case['source_sha256']}|{rep}"
                random.Random(int.from_bytes(
                    sha256(salt.encode("ascii")).digest(),"big")).shuffle(order)
                for arm in order:
                    start=time.perf_counter_ns()
                    result=run_arm(arm,records[rep],guards[arm])
                    duration_ms=(time.perf_counter_ns()-start)/1e6
                    if result!=(reference,expected_root):
                        raise ValueError("timed signed full witness/root disagreement "+arm)
                    if nor46.exact_convolution_prefix is not original_process_convolution:
                        raise RuntimeError("NOR46 global exact signed convolution changed")
                    samples[arm].append(duration_ms)
            for arm in ARMS:
                context=records[0]["context"]
                key=(case["source_sha256"],context.session_id,context.provider_id)
                if guards[arm].last[key]!=5:
                    raise ValueError("signed replay guard rejected monotone sequence")
            medians={arm:median(samples[arm]) for arm in ARMS}
            ratios={
                "always_native_over_always_GMP":medians[ARMS[1]]/medians[ARMS[0]],
                "frozen_selector_over_always_GMP":medians[ARMS[2]]/medians[ARMS[0]],
                "frozen_selector_over_always_native":medians[ARMS[2]]/medians[ARMS[1]]
            }
            rows.append({
                **label,"source_success":True,
                "exact_full_signed_vector_and_original_merkle_all_three":True,
                "frozen_public_selector_decision":decision,
                "trusted_public_max_coefficient_bitwidth":input_bitwidth,
                "original_full_merkle_root_sha256":expected_root,
                "full_native_reference_generation_ns_outside_arms":
                    full_reference_generation_ns,
                "full_reference_json_serialization_ns_outside_arms":full_json_ns,
                "full_reference_json_witness_bytes":len(full_json),
                "same_signed_envelope_bytes":records[0]["full_signed_wire_bytes"],
                "reduced_signed_payload_bytes":records[0]["reduced_witness_payload_bytes"],
                "prebuilt_fixture_signer_encode_ns":sum(
                    r["fixture_signer_encode_ns"] for r in records),
                "five_distinct_verifier_challenge_nonce_SHA256":[
                    r["verifier_nonce_sha256"] for r in records],
                "full_and_compressed_original_resource_bounds":bounds,
                "arm_medians_ms":medians,
                "five_interleaved_samples_ms":samples,
                **ratios
            })
            print("NOR57 FIRST CPU",ordinal,"/60",case["profile"],case["length"],
                  "trusted route",decision,
                  "native/GMP",round(ratios["always_native_over_always_GMP"],4),
                  "selector/GMP",round(ratios["frozen_selector_over_always_GMP"],4),
                  flush=True)
        except Exception as exc:
            rows.append({**label,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR57 FIRST SOURCE FAIL",ordinal,"/60",
                  type(exc).__name__,str(exc)[:200],flush=True)
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    success=[r for r in rows if r["source_success"]]
    chosen_native=[r for r in success if r["frozen_public_selector_decision"]==accept.NATIVE]
    chosen_gmp=[r for r in success if r["frozen_public_selector_decision"]==accept.GMP]
    totals=summarize(success)["sum_source_median_ms_by_arm"]
    ratio_of_sums=(totals[ARMS[2]]/totals[ARMS[0]]) if len(success)==60 else None
    original_unchanged=nor46.exact_convolution_prefix is original_process_convolution
    exact=(len(success)==60 and all(
        r["exact_full_signed_vector_and_original_merkle_all_three"] for r in success))
    gates={
        "all_60_selector_sum_source_median_over_GMP_under_0p95":(
            len(success)==60 and ratio_of_sums is not None and ratio_of_sums<.95),
        "eight_native_selected_native_over_GMP_median_under_0p85":(
            len(chosen_native)==8 and
            median(r["always_native_over_always_GMP"] for r in chosen_native)<.85),
        "fifty_two_GMP_selected_selector_over_GMP_median_under_1p10":(
            len(chosen_gmp)==52 and
            median(r["frozen_selector_over_always_GMP"] for r in chosen_gmp)<1.10),
        "all_60_signed_exact_full_vectors_original_roots_and_tamper_controls":exact,
        "first_source_60_348_old_excluded_5_repeats_fixed_route_caps_archive":(
            len(success)==60 and fixture["public_only"] and original_unchanged
            and len(chosen_native)==8 and len(chosen_gmp)==52
            and ids.isdisjoint(cohort.historical_ids())
            and all(all(len(r["five_interleaved_samples_ms"][arm])==REPEATS
                for arm in ARMS) for r in success)
            and all(0<r["full_and_compressed_original_resource_bounds"][
                    "original_selected_packed_bits"]<=nor46.MAX_PACKED_BITS
                    and 0<r["full_and_compressed_original_resource_bounds"][
                    "compressed_selected_packed_bits"]<=nor46.MAX_PACKED_BITS
                    and r["full_and_compressed_original_resource_bounds"][
                    "original_witness_max_bits"]<=nor46.MAX_COEFFICIENT_BITS
                    for r in success))
    }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:rss=None
    return {
        "schema":SCHEMA,"design_issue":269,
        "first_registration_sha":REGISTERED,
        "first_public_only_source_sha256":public_sha,
        "source_count":60,"source_success_count":len(success),
        "source_failure_count":60-len(success),
        "previous_original_source_exclusions":348,
        "repeat_count":REPEATS,"arms":list(ARMS),
        "source_first_SHA_frozen_before_witness_and_timing":True,
        "exact_same_signed_v2_wire_for_all_three_arms":True,
        "full_original_witness_and_merkle_root_each_arm":True,
        "same_common_signature_decode_full_caps_and_replay_pipeline":True,
        "witness_and_fixture_signing_outside_timed_verifier":True,
        "selector_ID":accept.RULE_ID,
        "registered_selector_rule":{
            "native_if_full_length_is_one_of":[769,1537],
            "native_if_trusted_stride_equals":4,
            "native_if_actual_public_max_coefficient_bit_length_equals":16,
            "otherwise":"GMP","never_reads_untrusted_witness_or_session_nonce":True},
        "original_reference_NOR56_signed_v2_parity_checked_before_timing":True,
        "selector_native_count":len(chosen_native),
        "selector_gmp_count":len(chosen_gmp),
        "nor46_global_exact_convolution_unchanged":original_unchanged,
        "python_version":sys.version.split()[0],
        "gmp_backend":fast_kernel.backend_metadata(),
        "cpu_model":platform.processor() or "unknown",
        "logical_cpu_count":os.cpu_count(),
        "maxrss_kib":rss,
        "primary_selector_ratio_of_sum_of_sixty_source_medians":ratio_of_sums,
        "primary_native_selected":summarize(chosen_native),
        "primary_GMP_selected":summarize(chosen_gmp),
        "overall":summarize(success),
        "by_length":{str(m):summarize([r for r in success if r["length"]==m])
                     for m in cohort.LENGTHS},
        "by_profile":{p:summarize([r for r in success if r["profile"]==p])
                      for p in cohort.PROFILES},
        "by_family":{f:summarize([r for r in success if r["family"]==f])
                     for f in cohort.FAMILIES},
        "secondary_ten_block_workload":workload_bootstrap(success),
        "original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "scope":"Transparent exact signed test-fixture witness, common authenticated pipeline, m3073 outside NOR45 signed grammar, no real key authority or provider honesty."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=REPEATS)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=run(args.public_file,args.repeats)
    raw=canonical(result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(raw)
    print("NOR57 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR57 ORIGINAL FIRST FIVE GATES",result["original_five_registered_gates"])
    print("NOR57 ORIGINAL FIRST ENGINEERING PASS",result["first_original_engineering_pass"])

if __name__=="__main__":main()
