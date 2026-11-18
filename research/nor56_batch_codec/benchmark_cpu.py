"""NOR56 first source-locked exact signed compact-v2 CPU experiment.

Only reads a SHA-frozen upstream 60-source public-only JSON. Measures four
complete-output/full-original-Merkle-root arms with five deterministic
source/repetition-interleaved repeats, and an untimed phase diagnostic after
whole-arm sampling. All first FAILs stay original, no re-selection.
"""
from __future__ import annotations
import argparse,json,os,platform,random,statistics,sys,time
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor51_exact_r_crossover import algebra as nor51
from research.nor52_sparse_inverse import algebra as support
from research.nor53_native_pack import fast_kernel
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import envelope as v1
from research.nor56_batch_codec import cohort,codec_v2,signed_v2,phase_diagnostic

REGISTERED=cohort.REGISTERED
SCHEMA="nor56.first_60_source_signed_compact_v2_cpu.v1"
REPEATS=5
ARMS=(
    "native_compressed_R_root",
    "signed_compact_v1_verifier_root",
    "signed_json_verifier_root",
    "signed_compact_v2_verifier_root"
)
SIGNED_ARMS=ARMS[1:]
ARM_FORMATS={
    ARMS[1]:"compact-v1",
    ARMS[2]:"json-decimal-v1",
    ARMS[3]:signed_v2.FORMAT_V2
}

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")
def med(vals):return statistics.median(vals) if vals else None

def summarize(rows):
    if not rows:return {"count":0}
    keys=(
       "signed_v2_over_signed_v1",
       "signed_v2_over_signed_json",
       "signed_v2_over_native",
       "signed_v1_over_native",
       "signed_json_over_native",
       "signed_v2_complete_wire_over_signed_v1_complete_wire",
       "signed_v2_complete_wire_over_signed_json_complete_wire"
    )
    return {
        "count":len(rows),
        **{"median_"+k:med(r[k] for r in rows) for k in keys},
        "median_arm_ms":{name:med(r["arm_medians_ms"][name] for r in rows)
                         for name in ARMS},
        "median_signed_v2_wire_bytes":med(
            r["signed_wire_bytes"][ARMS[3]] for r in rows),
        "median_signed_v1_wire_bytes":med(
            r["signed_wire_bytes"][ARMS[1]] for r in rows),
        "median_full_JSON_witness_payload_bytes":med(
            r["reference_full_JSON_witness_bytes"] for r in rows)
    }

def secondary_ten_block_bootstrap(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["replicate"],r["profile"])].append(r)
    if len(groups)!=10 or any(len(v)!=6 for v in groups.values()):
        return {"status":"incomplete ten-block original panel"}
    keys=sorted(groups)
    totals={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    comparable={}
    for i,a in enumerate(ARMS):
        if a==ARMS[1]:continue
        rng=random.Random(202611560000+i)
        rs=[]
        for _ in range(4096):
            sample=[r for b in (rng.choice(keys) for _ in keys) for r in groups[b]]
            rs.append(sum(r["arm_medians_ms"][a] for r in sample)/
                      sum(r["arm_medians_ms"][ARMS[1]] for r in sample))
        rs.sort()
        comparable[a]={
            "whole_workload_ratio_to_signed_compact_v1":totals[a]/totals[ARMS[1]],
            "descriptive_95pct_ten_block_bootstrap":[rs[102],rs[3993]]
        }
    return {
        "source_median_total_ms":totals,
        "original_source_blocks":10,
        "bootstrap_repeats":4096,
        "descriptive_comparisons_to_signed_compact_v1":comparable
    }

def make_signed_batch(reference,source,source_sha,stride,root):
    groups={}
    for arm in SIGNED_ARMS:
        fmt=ARM_FORMATS[arm]
        records=[]
        for i in range(REPEATS):
            sid=f"NOR56-{source_sha[:18]}-{fmt}"
            if fmt==signed_v2.FORMAT_V2:
                context=signed_v2.make_context_v2(
                    source=source,source_sha256=source_sha,stride=stride,
                    expected_root_sha256=root,session_id=sid,sequence=i+1)
            else:
                context=v1.make_context(
                    source=source,source_sha256=source_sha,stride=stride,
                    expected_root_sha256=root,payload_format=fmt,
                    session_id=sid,sequence=i+1)
            start=time.perf_counter_ns()
            wire=(signed_v2.make_signed_envelope_v2(reference,context)
                  if fmt==signed_v2.FORMAT_V2
                  else v1.make_signed_envelope(reference,context))
            prep_ns=time.perf_counter_ns()-start
            header,payload,signature,unsigned=v1.inspect_header(wire)
            expected=(signed_v2.expected_header(context,payload)
                      if fmt==signed_v2.FORMAT_V2
                      else v1._expected_header(context,payload))
            if header!=expected or len(signature)!=64:
                raise ValueError("before timing noncanonical signed envelope version")
            records.append({
                "wire":wire,"context":context,
                "payload_bytes":len(payload),
                "signed_wire_bytes":len(wire),
                "signer_encoding_and_signing_ns":prep_ns,
                "nonce_sha256":sha256(bytes.fromhex(context.nonce)).hexdigest()
            })
        groups[arm]=records
    return groups

def run_arm(name,source,reference,stride,signed,repetition,guards):
    if name==ARMS[0]:return support.native_compressed_root(source,stride)
    record=signed[name][repetition]
    if name==ARMS[3]:
        return signed_v2.verify_signed_envelope_v2(
            record["wire"],record["context"],guards[name])
    if name in ARMS[1:3]:
        return v1.verify_signed_envelope(
            record["wire"],record["context"],guards[name])
    raise ValueError("unknown first NOR56 arm")

def run(public_file:Path,repeats=REPEATS):
    if repeats!=REPEATS:
        raise ValueError("first original NOR56 must have five repeats")
    fixture,fixture_sha=cohort.read_fixture(public_file)
    ids={case["source_sha256"] for case in fixture["cases"]}
    assert len(ids)==60 and ids.isdisjoint(cohort.historical_ids())
    global_before=nor46.exact_convolution_prefix
    rows=[]
    for n,case in enumerate(fixture["cases"],1):
        label={k:case[k] for k in (
            "source_sha256","replicate","length","profile","family",
            "stride","coefficient_magnitude_bits")}
        try:
            f=nor51.normalized(tuple(int(x) for x in case["source_coefficients_decimal"]))
            if len(f)!=case["length"]:
                raise ValueError("source length mismatch")
            start=time.perf_counter_ns()
            reference=nor51.native_single_inverse(f)
            reference_generation_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            json_reference=producer.canonical_witness_bytes(reference)
            json_reference_ns=time.perf_counter_ns()-start
            root=nor51.root(reference)
            bounds=support.original_and_compressed_resource_sizes(
                f,reference,case["stride"])
            signed=make_signed_batch(
                reference,f,case["source_sha256"],case["stride"],root)
            # Pre-time exact outputs using distinct replay guards so first
            # timed nonce/sequence is still unused when it is measured.
            pre_guards={a:v1.ReplayGuard() for a in SIGNED_ARMS}
            for arm in ARMS:
                vals,verified_root=run_arm(
                    arm,f,reference,case["stride"],signed,0,pre_guards)
                if vals!=reference or verified_root!=root:
                    raise ValueError("pre-timing full output/root parity failure "+arm)
            guards={a:v1.ReplayGuard() for a in SIGNED_ARMS}
            samples={a:[] for a in ARMS}
            for rep in range(REPEATS):
                order=list(ARMS)
                seed=int.from_bytes(sha256(
                    f"NOR56/ARM-ORDER/v1|{case['source_sha256']}|{rep}".encode("ascii")
                ).digest(),"big")
                random.Random(seed).shuffle(order)
                for arm in order:
                    begin=time.perf_counter_ns()
                    vals,verified_root=run_arm(
                        arm,f,reference,case["stride"],signed,rep,guards)
                    elapsed=(time.perf_counter_ns()-begin)/1e6
                    if vals!=reference or verified_root!=root:
                        raise ValueError("timed full exact output/root mismatch "+arm)
                    if nor46.exact_convolution_prefix is not global_before:
                        raise RuntimeError("NOR46 process global convolution mutated")
                    samples[arm].append(elapsed)
            for arm in SIGNED_ARMS:
                context=signed[arm][0]["context"]
                accepted=guards[arm].last[(
                    case["source_sha256"],context.session_id,context.provider_id)]
                if accepted!=5:
                    raise ValueError("signed verifier session nonce/sequence gate failed")
            durations={a:med(samples[a]) for a in ARMS}
            wire={a:signed[a][0]["signed_wire_bytes"] for a in SIGNED_ARMS}
            payload={a:signed[a][0]["payload_bytes"] for a in SIGNED_ARMS}
            ratios={
                "signed_v2_over_signed_v1":durations[ARMS[3]]/durations[ARMS[1]],
                "signed_v2_over_signed_json":durations[ARMS[3]]/durations[ARMS[2]],
                "signed_v2_over_native":durations[ARMS[3]]/durations[ARMS[0]],
                "signed_v1_over_native":durations[ARMS[1]]/durations[ARMS[0]],
                "signed_json_over_native":durations[ARMS[2]]/durations[ARMS[0]],
                "signed_v2_complete_wire_over_signed_v1_complete_wire":
                    wire[ARMS[3]]/wire[ARMS[1]],
                "signed_v2_complete_wire_over_signed_json_complete_wire":
                    wire[ARMS[3]]/wire[ARMS[2]]
            }
            # The following separate stage replay is strictly after ALL
            # first registered timed samples for this source.
            phases={}
            for arm in SIGNED_ARMS:
                record=signed[arm][0]
                phases[arm]=phase_diagnostic.profile_one(
                    record["wire"],record["context"],ARM_FORMATS[arm],
                    reference,root)
            row={
                **label,"source_success":True,
                "all_four_full_signed_vector_and_original_merkle_roots_equal":True,
                "full_original_merkle_root_sha256":root,
                "reference_full_native_generation_ns_outside_arms":
                    reference_generation_ns,
                "reference_full_JSON_witness_bytes":len(json_reference),
                "reference_full_JSON_serialization_ns_outside_arms":
                    json_reference_ns,
                "signed_wire_bytes":wire,
                "payload_bytes":payload,
                "prebuilt_signer_encode_and_sign_ns":{
                    a:sum(v["signer_encoding_and_signing_ns"] for v in signed[a])
                    for a in SIGNED_ARMS},
                "nonce_digest_sha256":{
                    a:[v["nonce_sha256"] for v in signed[a]] for a in SIGNED_ARMS},
                "original_full_and_reduced_caps":bounds,
                "arm_medians_ms":durations,
                "five_interleaved_samples_ms":samples,
                "secondary_untimed_six_phase_ledger_ns":phases,
                **ratios
            }
            rows.append(row)
            print("NOR56 FIRST CPU",n,"/60",case["profile"],case["length"],
                  "v2/v1",round(ratios["signed_v2_over_signed_v1"],4),
                  "v2/json",round(ratios["signed_v2_over_signed_json"],4),
                  "wire-v2/v1",round(
                      ratios["signed_v2_complete_wire_over_signed_v1_complete_wire"],4),
                  flush=True)
        except Exception as exc:
            rows.append({**label,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR56 FIRST SOURCE FAIL",n,"/60",
                  type(exc).__name__,str(exc)[:180],flush=True)
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    valid=[r for r in rows if r["source_success"]]
    long=[r for r in valid if r["length"]==3073 and r["stride"]>1]
    sparse=[r for r in valid if r["stride"]>1]
    long_s=summarize(long)
    sparse_s=summarize(sparse)
    phases_ok=(len(valid)==60 and all(
        all(r["secondary_untimed_six_phase_ledger_ns"][a]["full_output_and_root_equal"] and
            r["secondary_untimed_six_phase_ledger_ns"][a]["signature_and_exact_identity_pass"]
            and r["secondary_untimed_six_phase_ledger_ns"][a]["phase_total_ns"]==
                sum(r["secondary_untimed_six_phase_ledger_ns"][a][k]
                    for k in phase_diagnostic.PHASES)
            for a in SIGNED_ARMS)
        for r in valid))
    reentrant=nor46.exact_convolution_prefix is global_before
    gates={
        "sparse_m3073_signed_v2_over_v1_under_0p92":(
            len(long)==16 and long_s["median_signed_v2_over_signed_v1"]<.92),
        "sparse_m3073_signed_v2_over_JSON_under_0p82":(
            len(long)==16 and long_s["median_signed_v2_over_signed_json"]<.82),
        "all_48_sparse_signed_v2_wire_over_v1_wire_under_1p05":(
            len(sparse)==48 and
            sparse_s["median_signed_v2_complete_wire_over_signed_v1_complete_wire"]<1.05),
        "all_60_exact_full_roots_and_signed_canonical_attack_controls":(
            len(valid)==60 and all(
                r["all_four_full_signed_vector_and_original_merkle_roots_equal"]
                for r in valid)),
        "first_source_60_288_excluded_five_repeats_caps_reentrant_archive":(
            len(valid)==60 and fixture["public_only"] and reentrant and
            ids.isdisjoint(cohort.historical_ids()) and phases_ok and
            all(all(len(r["five_interleaved_samples_ms"][a])==5 for a in ARMS)
                for r in valid) and
            all(0<r["original_full_and_reduced_caps"]["original_selected_packed_bits"]
                    <=nor46.MAX_PACKED_BITS and
                0<r["original_full_and_reduced_caps"]["compressed_selected_packed_bits"]
                    <=nor46.MAX_PACKED_BITS and
                r["original_full_and_reduced_caps"]["original_witness_max_bits"]
                    <=nor46.MAX_COEFFICIENT_BITS
                for r in valid))
    }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:rss=None
    return {
        "schema":SCHEMA,
        "design_issue":267,
        "registration_first_sha":REGISTERED,
        "first_source_public_only_SHA256":fixture_sha,
        "source_count":60,"source_success_count":len(valid),
        "source_failure_count":60-len(valid),
        "excluded_previous_first_source_ID_count":288,
        "repeat_count":REPEATS,"arms":list(ARMS),
        "prebuilt_fixture_signature_and_witness_outside_timed_verifier":True,
        "full_signed_integer_output_and_Merkle_each_timed_arm":True,
        "signed_protocol_v2_signature_domain_separated":True,
        "source_only_SHA_frozen_before_timed_CPU":True,
        "untimed_phase_diagnostics_after_all_four_arm_timings":True,
        "nor46_process_global_unchanged":reentrant,
        "GMP_metadata":fast_kernel.backend_metadata(),
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "cpu_model":platform.processor() or "unknown",
        "maxrss_kib":rss,
        "overall":summarize(valid),
        "primary_sixteen_sparse_m3073":long_s,
        "primary_48_sparse_wire":sparse_s,
        "by_length":{str(m):summarize([r for r in valid if r["length"]==m])
                     for m in cohort.LENGTHS},
        "by_profile":{p:summarize([r for r in valid if r["profile"]==p])
                      for p in cohort.PROFILES},
        "by_family":{f:summarize([r for r in valid if r["family"]==f])
                     for f in cohort.FAMILIES},
        "secondary_source_median_ten_block_bootstrap":
            secondary_ten_block_bootstrap(valid),
        "original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "limitations":"Exact transparent synthetic formal-series witnesses with publicly known Ed25519 test fixture key. m3073 outside NOR45 signed-native grammar; network/provider truth and real key protection external."
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
    print("NOR56 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR56 FIRST ORIGINAL FIVE REGISTERED GATES",result["original_five_registered_gates"])
    print("NOR56 FIRST ORIGINAL ENGINEERING PASS",result["first_original_engineering_pass"])

if __name__=="__main__":
    main()
