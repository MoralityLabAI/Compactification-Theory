"""NOR58 first source-frozen 60-source same-wire static indexed Merkle CPU.

The original first public-only fixture MUST arrive from a separate upstream
CI job. Three exact signed-v2 GMP identity verifier arms run five interleaved
whole-operation samples each, return the original full signed integer vector
and NOR45 Merkle root. All template construction uses only trusted (m,s).
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor45_authorized_program import independent_execution_verifier as nor45
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as nor51
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import signed_v2
from research.nor58_static_merkle import cohort,template,verify

REGISTERED=cohort.REGISTERED
SCHEMA="nor58.first_60_source_authenticated_static_merkle_cpu.v1"
REPEATS=5
ARMS=(
    "original_signed_GMP_full_Merkle",
    "warm_static_template_signed_GMP_full_Merkle",
    "cold_static_template_signed_GMP_full_Merkle"
)
MODE_BY_ARM={
    ARMS[0]:verify.ORIGINAL,
    ARMS[1]:verify.WARM,
    ARMS[2]:verify.COLD
}

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")
def median(vals):
    return statistics.median(vals) if vals else None

def summarize(rows):
    if not rows:return {"count":0}
    fields=("warm_over_original","cold_over_original","cold_over_warm",
            "template_cost_amortized_over_five_warm_over_original")
    return {
        "count":len(rows),
        **{"median_sourcewise_"+field:median(r[field] for r in rows)
           for field in fields},
        "median_ms_by_arm":{arm:median(r["arm_medians_ms"][arm] for r in rows)
                            for arm in ARMS},
        "sum_source_median_ms_by_arm":{
            arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS},
        "median_template_build_ms":median(r["template_build_ms_outside_arms"] for r in rows),
        "median_original_root_only_ns":median(
            r["untimed_root_only_original_ns"] for r in rows),
        "median_template_root_only_ns":median(
            r["untimed_root_only_template_ns"] for r in rows)
    }

def ten_block_bootstrap(rows):
    blocks=defaultdict(list)
    for r in rows:
        blocks[(r["replicate"],r["profile"])].append(r)
    if len(blocks)!=10 or any(len(v)!=6 for v in blocks.values()):
        return {"status":"incomplete original ten-block 60-source panel"}
    keys=sorted(blocks)
    sums={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    output={}
    for j,arm in enumerate(ARMS[1:]):
        rng=random.Random(202611580000+j)
        draws=[]
        for _ in range(4096):
            selected=[x for b in (rng.choice(keys) for _ in keys)
                      for x in blocks[b]]
            draws.append(sum(r["arm_medians_ms"][arm] for r in selected)/
                         sum(r["arm_medians_ms"][ARMS[0]] for r in selected))
        draws.sort()
        output[arm]={
            "ratio_of_sum_of_sixty_source_median_wall_ms":sums[arm]/sums[ARMS[0]],
            "descriptive_ten_block_95pct_interval":[draws[102],draws[3993]]
        }
    return {
        "block_count":10,"bootstrap_repeats":4096,
        "source_median_total_ms":sums,
        "secondary_descriptive_vs_original":output
    }

def prebuild_geometry_templates():
    cache={}
    ledger={}
    for m in cohort.LENGTHS:
        for s in (1,4,8):
            started=time.perf_counter_ns()
            result=template.build_template(m,s)
            ns=time.perf_counter_ns()-started
            key=f"m{m}-s{s}"
            cache[(m,s)]=result
            ledger[key]={
                "m":m,"stride":s,
                "prebuilt_source_geometry_only":True,
                "build_ns_outside_all_arm_timings":ns,
                "template_SHA256":result.template_sha256,
                "canonical_template_bytes":result.canonical_template_bytes,
                "static_node_count":result.static_node_count,
                "static_zero_leaf_count":result.static_zero_leaf_count,
                "static_padding_leaf_count":result.static_padding_leaf_count,
                "supported_dynamic_leaf_count":len(result.dynamic_leaf_indices)
            }
    assert len(cache)==len(ledger)==9
    return cache,ledger

def prebuild_signed_records(reference,source,source_sha,stride,expected_root):
    records=[]
    for rep in range(REPEATS):
        context=signed_v2.make_context_v2(
            source=source,source_sha256=source_sha,stride=stride,
            expected_root_sha256=expected_root,
            session_id=f"NOR58-SIGNED-{source_sha[:24]}",
            sequence=rep+1)
        start=time.perf_counter_ns()
        wire=signed_v2.make_signed_envelope_v2(reference,context)
        signer_ns=time.perf_counter_ns()-start
        head,payload,sig,unsigned=v1.inspect_header(wire)
        if head!=signed_v2.expected_header(context,payload) or len(sig)!=64:
            raise ValueError("original signed compact-v2 header mismatch before timing")
        records.append({
            "wire":wire,"context":context,"payload_bytes":len(payload),
            "signed_wire_bytes":len(wire),
            "signer_encode_and_sign_ns_outside_arms":signer_ns,
            "verifier_nonce_sha256":sha256(bytes.fromhex(context.nonce)).hexdigest()
        })
    if len({r["verifier_nonce_sha256"] for r in records})!=REPEATS:
        raise ValueError("duplicate verifier-issued challenge nonce")
    return records

def run_arm(arm,record,guard,static_template):
    return verify.verify_same_signed_witness(
        record["wire"],record["context"],guard,MODE_BY_ARM[arm],
        static_template if arm==ARMS[1] else None)

def run(public_file:Path,repeats:int=REPEATS):
    if repeats!=REPEATS:
        raise ValueError("first NOR58 requires exactly five signed whole-arm repeats")
    fixture,fixture_sha=cohort.read_fixture(public_file)
    first_ids={r["source_sha256"] for r in fixture["cases"]}
    assert len(first_ids)==60 and first_ids.isdisjoint(cohort.historical_ids())
    globals_before=nor46.exact_convolution_prefix
    templates,geometry_ledger=prebuild_geometry_templates()
    rows=[]
    for ordinal,case in enumerate(fixture["cases"],start=1):
        label={key:case[key] for key in (
            "source_sha256","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits")}
        try:
            f=nor51.normalized(tuple(int(x) for x in
                          case["source_coefficients_decimal"]))
            if len(f)!=case["length"]:
                raise ValueError("registered full public source length mismatch")
            started=time.perf_counter_ns()
            witness=nor51.native_single_inverse(f)
            producer_ns=time.perf_counter_ns()-started
            started=time.perf_counter_ns()
            reference_json=producer.canonical_witness_bytes(witness)
            serialization_ns=time.perf_counter_ns()-started
            expected_root=nor45.independently_computed_root(witness)
            bounds=support.original_and_compressed_resource_sizes(
                f,witness,case["stride"])
            tpl=templates[(len(f),case["stride"])]
            plan=geometry_ledger[f"m{len(f)}-s{case['stride']}"]
            if tpl.root(witness)!=expected_root:
                raise ValueError("pretime static indexed full NOR45 Merkle root mismatch")
            records=prebuild_signed_records(
                witness,f,case["source_sha256"],case["stride"],expected_root)
            # Previous original NOR56 signed v2 is independently checked
            # with its own fresh guard BEFORE any first timed samples.
            if signed_v2.verify_signed_envelope_v2(
                records[0]["wire"],records[0]["context"],
                v1.ReplayGuard())!=(witness,expected_root):
                raise ValueError("prior NOR56 signed exact source/root disagreement")
            pre_guards={a:v1.ReplayGuard() for a in ARMS}
            for a in ARMS:
                if run_arm(a,records[0],pre_guards[a],tpl)!=(witness,expected_root):
                    raise ValueError("pretime full signed integer/root mismatch "+a)
            guards={a:v1.ReplayGuard() for a in ARMS}
            samples={a:[] for a in ARMS}
            for rep in range(REPEATS):
                order=list(ARMS)
                token=f"NOR58/ARM-ORDER/v1|{case['source_sha256']}|{rep}"
                seed=int.from_bytes(sha256(token.encode("ascii")).digest(),"big")
                random.Random(seed).shuffle(order)
                for a in order:
                    started=time.perf_counter_ns()
                    actual=run_arm(a,records[rep],guards[a],tpl)
                    elapsed_ms=(time.perf_counter_ns()-started)/1e6
                    if actual!=(witness,expected_root):
                        raise ValueError("first timed signed complete witness/root mismatch "+a)
                    if nor46.exact_convolution_prefix is not globals_before:
                        raise RuntimeError("NOR46 global signed exact convolution was mutated")
                    samples[a].append(elapsed_ms)
            for a in ARMS:
                context=records[0]["context"]
                replay_key=(case["source_sha256"],context.session_id,context.provider_id)
                if guards[a].last[replay_key]!=5:
                    raise ValueError("original signed verifier nonce/session did not advance five times")
            med={a:median(samples[a]) for a in ARMS}
            original=med[ARMS[0]]
            ratios={
                "warm_over_original":med[ARMS[1]]/original,
                "cold_over_original":med[ARMS[2]]/original,
                "cold_over_warm":med[ARMS[2]]/med[ARMS[1]],
                "template_cost_amortized_over_five_warm_over_original":
                   (5*med[ARMS[1]]+plan["build_ns_outside_all_arm_timings"]/1e6)/
                   (5*original)
            }
            # Additional descriptive root-only times are strictly AFTER
            # all three signed whole-operation timing arms.
            start=time.perf_counter_ns()
            baseline_root=nor45.independently_computed_root(witness)
            baseline_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            template_root=tpl.root(witness)
            tmpl_ns=time.perf_counter_ns()-start
            if baseline_root!=template_root or baseline_root!=expected_root:
                raise ValueError("posttime independent original root parity failed")
            rows.append({
                **label,"source_success":True,
                "all_three_signed_full_exact_vectors_original_roots_equal":True,
                "reference_full_original_merkle_root_sha256":expected_root,
                "reference_full_native_generation_ns_outside_arms":producer_ns,
                "reference_json_serialization_ns_outside_arms":serialization_ns,
                "full_reference_decimal_JSON_bytes":len(reference_json),
                "signed_wire_bytes":records[0]["signed_wire_bytes"],
                "signed_reduced_payload_bytes":records[0]["payload_bytes"],
                "prebuilt_five_fixture_signer_encode_ns_outside_arms":
                    sum(record["signer_encode_and_sign_ns_outside_arms"]
                        for record in records),
                "five_distinct_verifier_nonce_SHA256":[
                    record["verifier_nonce_sha256"] for record in records],
                "template_geometry_only_key":f"m{len(f)}-s{case['stride']}",
                "template_SHA256":tpl.template_sha256,
                "template_build_ms_outside_arms":
                    plan["build_ns_outside_all_arm_timings"]/1e6,
                "template_canonical_bytes":tpl.canonical_template_bytes,
                "template_static_node_count":tpl.static_node_count,
                "template_dynamic_supported_leaf_count":len(tpl.dynamic_leaf_indices),
                "original_full_signed_resource_bounds":bounds,
                "five_interleaved_samples_ms":samples,
                "arm_medians_ms":med,
                "untimed_root_only_original_ns":baseline_ns,
                "untimed_root_only_template_ns":tmpl_ns,
                **ratios
            })
            print("NOR58 FIRST CPU",ordinal,"/60",case["profile"],case["length"],
                  "warm/orig",round(ratios["warm_over_original"],4),
                  "cold/orig",round(ratios["cold_over_original"],4),flush=True)
        except Exception as exc:
            rows.append({**label,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR58 FIRST SOURCE FAIL",ordinal,"/60",
                  type(exc).__name__,str(exc)[:180],flush=True)
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={m:20 for m in cohort.LENGTHS}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    success=[r for r in rows if r["source_success"]]
    sparse_long=[r for r in success if r["stride"]>1 and r["length"]==3073]
    dense=[r for r in success if r["stride"]==1]
    all_summary=summarize(success)
    sparse_summary=summarize(sparse_long)
    dense_summary=summarize(dense)
    all60_sums=all_summary["sum_source_median_ms_by_arm"] if len(success)==60 else {}
    ratio_of_sums=(
        all60_sums[ARMS[1]]/all60_sums[ARMS[0]] if len(success)==60 else None)
    same_globals=nor46.exact_convolution_prefix is globals_before
    exact=(len(success)==60 and all(
        r["all_three_signed_full_exact_vectors_original_roots_equal"] for r in success))
    gates={
        "sparse_m3073_warm_over_original_under_0p93":(
            len(sparse_long)==16 and
            sparse_summary["median_sourcewise_warm_over_original"]<.93),
        "all_60_sum_source_medians_warm_over_original_under_0p95":(
            len(success)==60 and ratio_of_sums is not None and ratio_of_sums<.95),
        "twelve_dense_cold_over_original_under_1p20":(
            len(dense)==12 and dense_summary["median_sourcewise_cold_over_original"]<1.20),
        "all_60_full_signed_vectors_roots_and_hostile_protocol_controls":exact,
        "first_60_source_468_excluded_repeats_static_template_caps_archive":(
            len(success)==60 and fixture["public_only"] and same_globals
            and first_ids.isdisjoint(cohort.historical_ids()) and len(templates)==9
            and all(all(len(r["five_interleaved_samples_ms"][a])==REPEATS for a in ARMS)
                    for r in success)
            and all(0<r["original_full_signed_resource_bounds"][
                        "original_selected_packed_bits"]<=nor46.MAX_PACKED_BITS
                    and 0<r["original_full_signed_resource_bounds"][
                        "compressed_selected_packed_bits"]<=nor46.MAX_PACKED_BITS
                    and r["original_full_signed_resource_bounds"][
                        "original_witness_max_bits"]<=nor46.MAX_COEFFICIENT_BITS
                    for r in success))
    }
    try:
        import resource
        maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:maxrss=None
    return {
        "schema":SCHEMA,"design_issue":274,
        "first_registration_sha":REGISTERED,
        "first_public_only_source_sha256":fixture_sha,
        "source_count":60,"source_success_count":len(success),
        "source_failure_count":60-len(success),
        "historical_original_first_source_ID_exclusions":468,
        "repeats":REPEATS,"signed_full_root_arms":list(ARMS),
        "all_arms_receive_identical_signed_v2_wire_and_context":True,
        "all_arms_check_original_full_caps_GMP_identity_and_full_root":True,
        "warm_template_only_trusted_geometry_no_dynamic_hash_cache":True,
        "source_only_SHA_frozen_upstream_of_first_CPU":True,
        "reference_and_fixture_signing_outside_arm_timings":True,
        "template_built_outside_arm_timings_for_warm":True,
        "cold_template_build_charged_inside_arm":True,
        "root_only_diagnostics_after_signed_timed_arms":True,
        "nor46_global_exact_signed_convolution_unchanged":same_globals,
        "geometry_template_ledger":geometry_ledger,
        "gmp_backend":fast_kernel.backend_metadata(),
        "cpu_model":platform.processor() or "unknown",
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),"maxrss_kib":maxrss,
        "primary_sparse_m3073":sparse_summary,
        "primary_twelve_dense_cold":dense_summary,
        "primary_all_sixty_ratio_of_sum_of_source_medians":ratio_of_sums,
        "overall":all_summary,
        "by_length":{str(m):summarize([r for r in success if r["length"]==m])
                     for m in cohort.LENGTHS},
        "by_profile":{p:summarize([r for r in success if r["profile"]==p])
                      for p in cohort.PROFILES},
        "by_family":{f:summarize([r for r in success if r["family"]==f])
                     for f in cohort.FAMILIES},
        "secondary_ten_block_descriptive_bootstrap":ten_block_bootstrap(success),
        "first_original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "limitations":"Exact synthetic signed integer witness; deterministic publicly known Ed25519 research test private seed, m3073 outside NOR45 signed grammar, no honest remote provider or ZK proof."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=REPEATS)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.public_file,a.repeats)
    raw=canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR58 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR58 FIRST ORIGINAL FIVE GATES",result["first_original_five_registered_gates"])
    print("NOR58 FIRST ORIGINAL ENGINEERING PASS",result["first_original_engineering_pass"])

if __name__=="__main__":
    main()
