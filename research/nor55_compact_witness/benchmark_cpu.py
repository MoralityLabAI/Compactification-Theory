"""NOR55 first source-frozen compact authenticated witness CPU benchmark.

The independent upstream source-only SHA fixture is required. Four timed arms
reconstruct the full signed output vector and original NOR45 Merkle root.
The same first-run source and frozen five engineering gates are immutable.
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
from research.nor55_compact_witness import cohort,codec,envelope

REGISTERED=cohort.REGISTERED
SCHEMA="nor55.first_60_signed_compact_witness_cpu.v1"
REPEATS=5
ARMS=("native_compressed_R_root","raw_fast_compressed_gmp_root",
      "signed_compact_verifier_root","signed_json_verifier_root")

def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")

def med(vals):
    return statistics.median(vals) if vals else None

def result_summary(rows):
    if not rows:return {"count":0}
    fields=(
        "raw_gmp_over_native",
        "signed_compact_over_native",
        "signed_compact_over_signed_json",
        "signed_compact_over_raw_gmp",
        "signed_json_over_raw_gmp",
        "compact_signed_wire_over_full_json_payload",
        "compact_signed_wire_over_signed_json_wire"
    )
    return {
        "count":len(rows),
        **{"median_"+field:med(r[field] for r in rows) for field in fields},
        "median_arm_ms":{a:med(r["arm_medians_ms"][a] for r in rows) for a in ARMS},
        "median_full_witness_json_bytes":med(r["json_witness_payload_bytes"] for r in rows),
        "median_compact_signed_wire_bytes":med(r["compact_signed_wire_bytes"] for r in rows)
    }

def secondary_blocks(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["replicate"],r["profile"])].append(r)
    if len(groups)!=10 or any(len(v)!=6 for v in groups.values()):
        return {"status":"incomplete source-matched ten-block cohort"}
    keys=sorted(groups)
    totals={arm:sum(r["arm_medians_ms"][arm] for r in rows) for arm in ARMS}
    results={}
    for i,arm in enumerate(ARMS[1:],start=1):
        rng=random.Random(202611550000+i)
        boot=[]
        for _ in range(4096):
            sample=[r for b in (rng.choice(keys) for _ in keys) for r in groups[b]]
            boot.append(sum(r["arm_medians_ms"][arm] for r in sample)/
                        sum(r["arm_medians_ms"][ARMS[0]] for r in sample))
        boot.sort()
        results[arm]={
            "source_median_workload_ratio_to_native":totals[arm]/totals[ARMS[0]],
            "descriptive_ten_block_95pct_interval":[boot[102],boot[3993]]
        }
    return {
        "sum_source_median_ms":totals,
        "bootstrap_repetitions":4096,
        "source_blocks":10,
        "secondary_matched_workload":results
    }

def arm_run(name,source,reference,stride,signed,repeat,guards):
    if name==ARMS[0]:
        return support.native_compressed_root(source,stride)
    if name==ARMS[1]:
        return fast_kernel.compressed_verified_root(source,reference,stride)
    if name in (ARMS[2],ARMS[3]):
        record=signed[name][repeat]
        return envelope.verify_signed_envelope(
            record["wire"],record["context"],guards[name])
    raise ValueError("unknown NOR55 frozen whole-operation arm")

def make_signed_batch(reference,source,source_sha,stride,root):
    signed={}
    for fmt,arm in (("compact-v1",ARMS[2]),("json-decimal-v1",ARMS[3])):
        batch=[]
        for repeat in range(REPEATS):
            # The verifier independently issues a fresh 256-bit challenge
            # prior to timing. Research fixture signing happens outside arms.
            ctx=envelope.make_context(
                source=source,source_sha256=source_sha,
                stride=stride,expected_root_sha256=root,
                payload_format=fmt,session_id=f"NOR55-{source_sha[:20]}-{fmt}",
                sequence=repeat+1)
            start=time.perf_counter_ns()
            wire=envelope.make_signed_envelope(reference,ctx)
            prep_ns=time.perf_counter_ns()-start
            header,payload,signature,unsigned=envelope.inspect_header(wire)
            assert len(signature)==64 and header["sequence"]==repeat+1
            batch.append({
                "wire":wire,"context":ctx,
                "wire_bytes":len(wire),"payload_bytes":len(payload),
                "prebuilt_encode_and_fixture_sign_ns":prep_ns,
                "nonce_sha256":sha256(bytes.fromhex(ctx.nonce)).hexdigest()
            })
        signed[arm]=batch
    return signed

def run(public_file:Path,repeats:int=REPEATS):
    if repeats!=REPEATS:
        raise ValueError("original NOR55 requires five interleaved repetitions")
    fixture,source_sha=cohort.read_fixture(public_file)
    ids={case["source_sha256"] for case in fixture["cases"]}
    assert len(ids)==60 and ids.isdisjoint(cohort.earlier_source_ids())
    global_before=nor46.exact_convolution_prefix
    rows=[]
    for index,case in enumerate(fixture["cases"]):
        identity={k:case[k] for k in (
            "source_sha256","replicate","length","profile",
            "family","stride","coefficient_magnitude_bits")}
        try:
            f=nor51.normalized(tuple(int(x) for x in case["source_coefficients_decimal"]))
            if len(f)!=case["length"]:
                raise ValueError("registered full source length mismatch")
            start=time.perf_counter_ns()
            reference=nor51.native_single_inverse(f)
            reference_ns=time.perf_counter_ns()-start
            start=time.perf_counter_ns()
            full_json=producer.canonical_witness_bytes(reference)
            full_json_serialization_ns=time.perf_counter_ns()-start
            expected_root=nor51.root(reference)
            bounds=support.original_and_compressed_resource_sizes(
                f,reference,case["stride"])
            signed=make_signed_batch(reference,f,case["source_sha256"],
                                     case["stride"],expected_root)
            compact=signed[ARMS[2]][0]
            full=signed[ARMS[3]][0]
            if compact["payload_bytes"]>codec.MAX_PAYLOAD_BYTES:
                raise ValueError("original compact payload cap exceeded")
            # Pre-run all four arms on an independent replay guard, leaving
            # five real timed verifier-issued challenges unconsumed.
            pre_guards={ARMS[2]:envelope.ReplayGuard(),
                        ARMS[3]:envelope.ReplayGuard()}
            for name in ARMS:
                vals,root=arm_run(name,f,reference,case["stride"],signed,0,pre_guards)
                if vals!=reference or root!=expected_root:
                    raise ValueError("first pre-timing output/root parity "+name)
            guards={ARMS[2]:envelope.ReplayGuard(),
                    ARMS[3]:envelope.ReplayGuard()}
            samples={arm:[] for arm in ARMS}
            for repetition in range(REPEATS):
                shuffled=list(ARMS)
                token=f"NOR55/ARM-ORDER/v1|{case['source_sha256']}|{repetition}"
                random.Random(int.from_bytes(sha256(token.encode("ascii")).digest(),
                                             "big")).shuffle(shuffled)
                for name in shuffled:
                    start=time.perf_counter_ns()
                    vals,root=arm_run(name,f,reference,case["stride"],
                                      signed,repetition,guards)
                    ms=(time.perf_counter_ns()-start)/1e6
                    if vals!=reference or root!=expected_root:
                        raise ValueError("timed source/full-root mismatch "+name)
                    if nor46.exact_convolution_prefix is not global_before:
                        raise RuntimeError("NOR46 process-global convolution mutated")
                    samples[name].append(ms)
            for name in (ARMS[2],ARMS[3]):
                session_name=signed[name][0]["context"].session_id
                assert guards[name].last[(case["source_sha256"],
                       session_name,envelope.PROVIDER_ID)]==5
            medians={arm:med(samples[arm]) for arm in ARMS}
            ratios={
                "raw_gmp_over_native":medians[ARMS[1]]/medians[ARMS[0]],
                "signed_compact_over_native":medians[ARMS[2]]/medians[ARMS[0]],
                "signed_compact_over_signed_json":medians[ARMS[2]]/medians[ARMS[3]],
                "signed_compact_over_raw_gmp":medians[ARMS[2]]/medians[ARMS[1]],
                "signed_json_over_raw_gmp":medians[ARMS[3]]/medians[ARMS[1]],
                "compact_signed_wire_over_full_json_payload":
                    compact["wire_bytes"]/len(full_json),
                "compact_signed_wire_over_signed_json_wire":
                    compact["wire_bytes"]/full["wire_bytes"]
            }
            row={
                **identity,"source_success":True,
                "full_exact_integer_vector_and_independent_merkle_root_all_four":True,
                "reference_full_merkle_root_sha256":expected_root,
                "reference_full_inverse_generation_ns_outside_arms":reference_ns,
                "reference_full_JSON_serialization_ns_outside_arms":full_json_serialization_ns,
                "json_witness_payload_bytes":len(full_json),
                "compact_payload_bytes":compact["payload_bytes"],
                "compact_signed_wire_bytes":compact["wire_bytes"],
                "json_signed_wire_bytes":full["wire_bytes"],
                "independent_prebuilt_fixture_signing_ns":{
                    arm:sum(r["prebuilt_encode_and_fixture_sign_ns"] for r in signed[arm])
                    for arm in (ARMS[2],ARMS[3])},
                "verifier_challenge_nonce_sha256":{
                    arm:[r["nonce_sha256"] for r in signed[arm]]
                    for arm in (ARMS[2],ARMS[3])},
                "reference_signed_original_caps":bounds,
                "arm_medians_ms":medians,
                "five_interleaved_samples_ms":samples,
                **ratios
            }
            rows.append(row)
            print("NOR55 FIRST CPU",index+1,"/60",case["profile"],case["length"],
                  "signed compact/native",round(ratios["signed_compact_over_native"],4),
                  "signed compact/json",round(ratios["signed_compact_over_signed_json"],4),
                  "wire/full JSON",round(ratios["compact_signed_wire_over_full_json_payload"],4),
                  flush=True)
        except Exception as exc:
            rows.append({**identity,"source_success":False,
                         "failure_type":type(exc).__name__,
                         "failure_message":str(exc)[:500]})
            print("NOR55 FIRST SOURCE FAIL",index+1,"/60",
                  type(exc).__name__,str(exc)[:180],flush=True)
    success=[r for r in rows if r["source_success"]]
    assert len(rows)==60 and len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={n:20 for n in cohort.LENGTHS}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    sparse_long=[r for r in success if r["stride"]>1 and r["length"]==3073]
    sparse=[r for r in success if r["stride"]>1]
    sparse_long_s=result_summary(sparse_long)
    sparse_s=result_summary(sparse)
    all_exact=len(success)==60 and all(
        r["full_exact_integer_vector_and_independent_merkle_root_all_four"] for r in success)
    reentrant=nor46.exact_convolution_prefix is global_before
    gates={
        "sparse_m3073_signed_compact_verifier_over_native_under_0p85":(
            len(sparse_long)==16 and
            sparse_long_s["median_signed_compact_over_native"]<.85),
        "sparse_m3073_signed_compact_over_signed_json_under_0p85":(
            len(sparse_long)==16 and
            sparse_long_s["median_signed_compact_over_signed_json"]<.85),
        "all_48_sparse_signed_compact_wire_over_full_json_payload_under_0p55":(
            len(sparse)==48 and
            sparse_s["median_compact_signed_wire_over_full_json_payload"]<.55),
        "all_exact_full_roots_pinned_key_source_session_replay_tamper_controls":all_exact,
        "source_first_60_sha_228_disjoint_five_repeats_caps_reentrant_archive":(
            len(success)==60 and fixture["public_only"] and reentrant and
            ids.isdisjoint(cohort.earlier_source_ids()) and
            all(all(len(r["five_interleaved_samples_ms"][name])==5 for name in ARMS)
                for r in success) and
            all(0<r["reference_signed_original_caps"]["original_selected_packed_bits"]
                  <=nor46.MAX_PACKED_BITS and
                r["reference_signed_original_caps"]["original_witness_max_bits"]
                  <=nor46.MAX_COEFFICIENT_BITS for r in success))
    }
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:rss=None
    return {
        "schema":SCHEMA,"design_issue":265,
        "first_registration_sha":REGISTERED,
        "first_public_only_source_sha256":source_sha,
        "source_count":60,"source_success_count":len(success),
        "source_failure_count":60-len(success),
        "previous_excluded_first_source_count":228,
        "repeats":REPEATS,"four_timed_arms":list(ARMS),
        "public_only_SHA_upstream_of_first_CPU":True,
        "trusted_fixture_key_is_explicitly_insecure_for_deployment":True,
        "signature_and_canonical_witness_decode_in_authenticated_arm":True,
        "fixture_witness_generation_and_signing_outside_verifier_arms":True,
        "full_integer_vector_original_merkle_root_all_arms":True,
        "caps_and_replay_checked_at_original_full_length":True,
        "no_process_global_NOR46_monkeypatch":reentrant,
        "cpu_model":platform.processor() or "unknown",
        "python_version":sys.version.split()[0],
        "logical_cpu_count":os.cpu_count(),
        "maxrss_kib":rss,
        "gmp_backend":fast_kernel.backend_metadata(),
        "overall":result_summary(success),
        "primary_sparse_m3073":sparse_long_s,
        "primary_all_sparse_wire":sparse_s,
        "by_length":{str(n):result_summary([r for r in success if r["length"]==n])
                     for n in cohort.LENGTHS},
        "by_profile":{p:result_summary([r for r in success if r["profile"]==p])
                      for p in cohort.PROFILES},
        "by_family":{fam:result_summary([r for r in success if r["family"]==fam])
                     for fam in cohort.FAMILIES},
        "secondary_matched_source_median_workload":secondary_blocks(success),
        "first_original_five_registered_gates":gates,
        "first_original_engineering_pass":all(gates.values()),
        "source_records":rows,
        "limitations":"Synthetic signed fixture key publicly deterministic; compact transparent full witness, m3073 outside NOR45 signed-native grammar; local loopback and provider honesty are separately scoped."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=REPEATS)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    x=run(a.public_file,a.repeats)
    raw=canonical(x)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR55 FIRST ORIGINAL CPU RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR55 ORIGINAL FIRST FIVE GATES",x["first_original_five_registered_gates"])
    print("NOR55 ORIGINAL FIRST ENGINEERING PASS",x["first_original_engineering_pass"])

if __name__=="__main__":
    main()
