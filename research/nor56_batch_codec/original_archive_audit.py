"""NOR56 permanent read-only first source/CPU raw SHA and engineering FAIL auditor.

The first source and CPU JSON bytes are checked inside an immutable ZIP.
No source regeneration, inverse execution, Ed25519 signing or retiming occurs.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent/"results"
REGISTERED="b186ee7478f27230478b099d7091b25817839464"
FIRST_RUN=44296895730
SOURCE_ARTIFACT=14423891566
CPU_ARTIFACT=14424252022
CPU_BYTES=400679
CPU_SHA="2ad2c0e149ca9f2405f7883fbb768fc72f380a03d717c2b265883343a6d7429d"
SOURCE_BYTES=1473496
SOURCE_SHA="f1326f4ac9cac314564c96eb82008177f30280b38ab8972eab696f51996b421c"
ARMS=(
    "native_compressed_R_root",
    "signed_compact_v1_verifier_root",
    "signed_json_verifier_root",
    "signed_compact_v2_verifier_root"
)
SIGNED_ARMS=ARMS[1:]
PHASES=(
    "header_and_replay_ns","ed25519_signature_ns",
    "canonical_payload_decode_ns","full_source_witness_support_caps_ns",
    "signed_exact_GMP_convolution_ns","full_original_Merkle_root_ns"
)
GATES={
  "sparse_m3073_signed_v2_over_v1_under_0p92":False,
  "sparse_m3073_signed_v2_over_JSON_under_0p82":False,
  "all_48_sparse_signed_v2_wire_over_v1_wire_under_1p05":True,
  "all_60_exact_full_roots_and_signed_canonical_attack_controls":True,
  "first_source_60_288_excluded_five_repeats_caps_reentrant_archive":True
}
PRIMARY={
 "sparse_m3073_v2_over_v1":0.9782139098028163,
 "sparse_m3073_v2_over_JSON":0.8355006990337561,
 "sparse_48_v2_complete_wire_over_v1":0.9992231696183432
}

def audit():
    archive=HERE/"nor56_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor56_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor56.original_first_signed_v2_source_cpu_archive.v1"
    assert receipt["first_run"]==FIRST_RUN
    assert receipt["original_source_artifact_id"]==SOURCE_ARTIFACT
    assert receipt["original_cpu_artifact_id"]==CPU_ARTIFACT
    assert receipt["first_registration_sha"]==REGISTERED
    assert receipt["raw_cpu_bytes"]==CPU_BYTES and receipt["raw_cpu_sha256"]==CPU_SHA
    assert receipt["raw_source_bytes"]==SOURCE_BYTES and receipt["raw_source_sha256"]==SOURCE_SHA
    assert receipt["original_five_registered_gates"]==GATES
    assert receipt["first_original_engineering_pass"] is False
    assert receipt["original_primary"]==PRIMARY
    assert archive.stat().st_size==receipt["original_archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["original_archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor56_first_original_cpu.json",
            "nor56_first_original_60_public_only.json"}
        raw=z.read("nor56_first_original_cpu.json")
        public=z.read("nor56_first_original_60_public_only.json")
    assert len(raw)==CPU_BYTES and sha256(raw).hexdigest()==CPU_SHA
    assert len(public)==SOURCE_BYTES and sha256(public).hexdigest()==SOURCE_SHA
    x,fixture=json.loads(raw),json.loads(public)
    assert x["schema"]=="nor56.first_60_source_signed_compact_v2_cpu.v1"
    assert fixture["schema"]=="nor56.first_public_only_60_batch_table_sources.v1"
    assert x["registration_first_sha"]==fixture["first_registration_sha"]==REGISTERED
    assert x["first_source_public_only_SHA256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert x["excluded_previous_first_source_ID_count"]==288
    assert fixture["source_count"]==60 and fixture["excluded_original_source_IDs"]==288
    assert fixture["public_only"] and fixture["no_inverse_root_payload_signature_session_or_timing"]
    assert fixture["source_domain"]=="NOR56/SOURCE/v1"
    assert x["repeat_count"]==5 and x["arms"]==list(ARMS)
    assert x["prebuilt_fixture_signature_and_witness_outside_timed_verifier"]
    assert x["full_signed_integer_output_and_Merkle_each_timed_arm"]
    assert x["signed_protocol_v2_signature_domain_separated"]
    assert x["source_only_SHA_frozen_before_timed_CPU"]
    assert x["untimed_phase_diagnostics_after_all_four_arm_timings"]
    assert x["nor46_process_global_unchanged"]
    assert x["GMP_metadata"]["gmpy2_version"]=="2.3.2"
    assert x["original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is False
    cases=fixture["cases"]
    ids={r["source_sha256"] for r in cases}
    assert len(ids)==len(cases)==60
    rows=x["source_records"]
    assert len(rows)==60 and {r["source_sha256"] for r in rows}==ids
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert Counter(r["profile"] for r in rows)=={
        "s1-b2":12,"s4-b2":12,"s8-b2":12,"s4-b8":12,"s4-b16":12}
    assert Counter(r["family"] for r in rows)=={
        "signed_random":30,"alternating":30}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    by_id={r["source_sha256"]:r for r in cases}
    for row in rows:
        assert row["source_success"] and row["all_four_full_signed_vector_and_original_merkle_roots_equal"]
        inp=by_id[row["source_sha256"]]
        for key in ("replicate","length","profile","family","stride","coefficient_magnitude_bits"):
            assert row[key]==inp[key]
        assert len(row["full_original_merkle_root_sha256"])==64
        assert row["reference_full_native_generation_ns_outside_arms"]>=0
        assert row["reference_full_JSON_witness_bytes"]>0
        assert row["reference_full_JSON_serialization_ns_outside_arms"]>=0
        bounds=row["original_full_and_reduced_caps"]
        assert bounds["original_coefficient_count"]==row["length"]
        assert bounds["compressed_coefficient_count"]==(
            (row["length"]-1)//row["stride"]+1)
        assert 0<bounds["original_selected_packed_bits"]<=120_000_000
        assert 0<bounds["compressed_selected_packed_bits"]<=120_000_000
        assert bounds["original_witness_max_bits"]<=32768
        ms,samples=row["arm_medians_ms"],row["five_interleaved_samples_ms"]
        assert set(ms)==set(samples)==set(ARMS)
        for arm in ARMS:
            times=samples[arm]
            assert len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
            assert math.isclose(statistics.median(times),ms[arm],rel_tol=1e-12)
        wire=row["signed_wire_bytes"]
        payload=row["payload_bytes"]
        assert set(wire)==set(payload)==set(SIGNED_ARMS)
        for arm in SIGNED_ARMS:
            assert wire[arm]>payload[arm]>0
            assert row["prebuilt_signer_encode_and_sign_ns"][arm]>=0
        sha_per_arm=row["nonce_digest_sha256"]
        assert set(sha_per_arm)==set(SIGNED_ARMS)
        nonces=[h for a in SIGNED_ARMS for h in sha_per_arm[a]]
        assert len(nonces)==len(set(nonces))==15
        assert all(len(h)==64 for h in nonces)
        ratios={
            "signed_v2_over_signed_v1":ms[ARMS[3]]/ms[ARMS[1]],
            "signed_v2_over_signed_json":ms[ARMS[3]]/ms[ARMS[2]],
            "signed_v2_over_native":ms[ARMS[3]]/ms[ARMS[0]],
            "signed_v1_over_native":ms[ARMS[1]]/ms[ARMS[0]],
            "signed_json_over_native":ms[ARMS[2]]/ms[ARMS[0]],
            "signed_v2_complete_wire_over_signed_v1_complete_wire":
                wire[ARMS[3]]/wire[ARMS[1]],
            "signed_v2_complete_wire_over_signed_json_complete_wire":
                wire[ARMS[3]]/wire[ARMS[2]]
        }
        for key,value in ratios.items():
            assert math.isclose(row[key],value,rel_tol=1e-12)
        phases=row["secondary_untimed_six_phase_ledger_ns"]
        assert set(phases)==set(SIGNED_ARMS)
        for arm in SIGNED_ARMS:
            data=phases[arm]
            assert data["full_output_and_root_equal"]
            assert data["signature_and_exact_identity_pass"]
            assert data["original_full_length"]==row["length"]
            assert data["decoded_reduced_length"]==bounds["compressed_coefficient_count"]
            assert data["payload_bytes"]==payload[arm]
            assert data["signed_wire_bytes"]==wire[arm]
            assert all(type(data[k]) is int and data[k]>=0 for k in PHASES)
            assert data["phase_total_ns"]==sum(data[k] for k in PHASES)
    longest=[r for r in rows if r["length"]==3073 and r["stride"]>1]
    sparse=[r for r in rows if r["stride"]>1]
    assert len(longest)==16 and len(sparse)==48
    observed={
        "sparse_m3073_v2_over_v1":statistics.median(
            r["signed_v2_over_signed_v1"] for r in longest),
        "sparse_m3073_v2_over_JSON":statistics.median(
            r["signed_v2_over_signed_json"] for r in longest),
        "sparse_48_v2_complete_wire_over_v1":statistics.median(
            r["signed_v2_complete_wire_over_signed_v1_complete_wire"]
            for r in sparse)
    }
    for k,val in PRIMARY.items():
        assert math.isclose(observed[k],val,abs_tol=1e-12)
    assert observed["sparse_m3073_v2_over_v1"]>=.92
    assert observed["sparse_m3073_v2_over_JSON"]>=.82
    assert observed["sparse_48_v2_complete_wire_over_v1"]<1.05
    totals=x["secondary_source_median_ten_block_bootstrap"]["source_median_total_ms"]
    for a in ARMS:
        assert math.isclose(
            totals[a],sum(r["arm_medians_ms"][a] for r in rows),rel_tol=1e-12)
    return {
        "original_run":FIRST_RUN,"CPU_SHA256":CPU_SHA,
        "first_verdict":"FAIL (3/5)","sources":60,
        "registered_primary":observed
    }
if __name__=="__main__":
    print("NOR56 first original negative signed v2 evidence SHA audit",audit())
