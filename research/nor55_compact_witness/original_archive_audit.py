"""NOR55: read-only SHA and original first 4-of-5 engineering FAIL auditor.

The original primary CPU JSON, upstream source-only fixture and separate
two-process localhost TCP secondary are preserved byte-for-byte inside a
single source-controlled archive. Never rerun or retime first heldout sources.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent/"results"
REGISTERED="9f7fb42a2e0a8648db63301bed381cb7f3aa0dc2"
FIRST_RUN=44292743862
FIRST_CPU_ARTIFACT=14421997454
FIRST_SOURCE_ARTIFACT=14421873595
FIRST_SECONDARY_ARTIFACT=14422382237
CPU_BYTES=230272
CPU_SHA="98c45ed555349fcfd860def1975d571d0e84565e6cad976d30941cae3384b10e"
SOURCE_BYTES=1474567
SOURCE_SHA="d1e6b2f5f37edc37f4b996b42e10768b47b4c199ed274cf53643be34fddddd59"
LOOPBACK_BYTES=20128
LOOPBACK_SHA="45d06c36990cd1cd005e6a0c8478ddca57f52afe1fef168c149e45787063e761"
ARMS=("native_compressed_R_root","raw_fast_compressed_gmp_root",
      "signed_compact_verifier_root","signed_json_verifier_root")
AUTH_ARMS=(ARMS[2],ARMS[3])
GATES={
  "sparse_m3073_signed_compact_verifier_over_native_under_0p85":True,
  "sparse_m3073_signed_compact_over_signed_json_under_0p85":False,
  "all_48_sparse_signed_compact_wire_over_full_json_payload_under_0p55":True,
  "all_exact_full_roots_pinned_key_source_session_replay_tamper_controls":True,
  "source_first_60_sha_228_disjoint_five_repeats_caps_reentrant_archive":True
}
PRIMARY={
    "sparse_m3073_signed_compact_over_native":0.744645195351167,
    "sparse_m3073_signed_compact_over_signed_json":0.8514830908862547,
    "all_sparse_signed_compact_wire_over_full_json_payload":0.4128419517968882
}

def audit():
    archive=HERE/"nor55_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor55_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor55.first_original_60_source_signed_compact_archive.v1"
    assert receipt["first_run"]==FIRST_RUN
    assert receipt["first_cpu_artifact_id"]==FIRST_CPU_ARTIFACT
    assert receipt["first_source_artifact_id"]==FIRST_SOURCE_ARTIFACT
    assert receipt["first_secondary_TCP_artifact_id"]==FIRST_SECONDARY_ARTIFACT
    assert receipt["first_registration_sha"]==REGISTERED
    assert receipt["raw_cpu_bytes"]==CPU_BYTES and receipt["raw_cpu_sha256"]==CPU_SHA
    assert receipt["raw_source_bytes"]==SOURCE_BYTES and receipt["raw_source_sha256"]==SOURCE_SHA
    assert receipt["raw_secondary_bytes"]==LOOPBACK_BYTES
    assert receipt["raw_secondary_sha256"]==LOOPBACK_SHA
    assert receipt["first_original_five_registered_gates"]==GATES
    assert receipt["original_first_engineering_pass"] is False
    assert receipt["original_primary"]==PRIMARY
    assert archive.stat().st_size==receipt["archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor55_first_original_cpu.json",
            "nor55_first_original_60_public_only.json",
            "nor55_first_local_loopback_secondary.json"}
        cpu=z.read("nor55_first_original_cpu.json")
        public=z.read("nor55_first_original_60_public_only.json")
        tcp=z.read("nor55_first_local_loopback_secondary.json")
    assert len(cpu)==CPU_BYTES and sha256(cpu).hexdigest()==CPU_SHA
    assert len(public)==SOURCE_BYTES and sha256(public).hexdigest()==SOURCE_SHA
    assert len(tcp)==LOOPBACK_BYTES and sha256(tcp).hexdigest()==LOOPBACK_SHA
    x,fixture,secondary=json.loads(cpu),json.loads(public),json.loads(tcp)
    assert x["schema"]=="nor55.first_60_signed_compact_witness_cpu.v1"
    assert fixture["schema"]=="nor55.first_public_only_60_signed_compact_sources.v1"
    assert x["first_registration_sha"]==fixture["first_registration_sha"]==REGISTERED
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert x["previous_excluded_first_source_count"]==228
    assert fixture["source_count"]==60 and fixture["earlier_excluded_source_count"]==228
    assert fixture["public_only"] and fixture["no_inverse_root_witness_session_payload_or_timing"]
    assert x["repeats"]==5 and x["four_timed_arms"]==list(ARMS)
    assert x["public_only_SHA_upstream_of_first_CPU"] is True
    assert x["signature_and_canonical_witness_decode_in_authenticated_arm"] is True
    assert x["trusted_fixture_key_is_explicitly_insecure_for_deployment"] is True
    assert x["full_integer_vector_original_merkle_root_all_arms"] is True
    assert x["caps_and_replay_checked_at_original_full_length"] is True
    assert x["no_process_global_NOR46_monkeypatch"] is True
    assert x["gmp_backend"]["gmpy2_version"]=="2.3.2"
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is False
    cases=fixture["cases"]
    public_ids={r["source_sha256"] for r in cases}
    assert len(cases)==len(public_ids)==60
    rows=x["source_records"]
    assert len(rows)==60 and {r["source_sha256"] for r in rows}==public_ids
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert Counter(r["profile"] for r in rows)=={
       "s1-b2":12,"s4-b2":12,"s8-b2":12,"s4-b8":12,"s4-b16":12}
    assert Counter(r["family"] for r in rows)=={
       "signed_random":30,"alternating":30}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    by_id={c["source_sha256"]:c for c in cases}
    for row in rows:
        assert row["source_success"] and row["full_exact_integer_vector_and_independent_merkle_root_all_four"]
        core=by_id[row["source_sha256"]]
        for k in ("replicate","length","profile","family","stride",
                  "coefficient_magnitude_bits"):
            assert row[k]==core[k]
        assert len(row["reference_full_merkle_root_sha256"])==64
        assert row["reference_full_inverse_generation_ns_outside_arms"]>=0
        assert row["reference_full_JSON_serialization_ns_outside_arms"]>=0
        assert row["json_witness_payload_bytes"]>0
        assert row["compact_signed_wire_bytes"]>0
        assert row["json_signed_wire_bytes"]>=row["json_witness_payload_bytes"]
        assert row["compact_signed_wire_bytes"]>=row["compact_payload_bytes"]
        assert row["compact_signed_wire_bytes"]<=8_500_000
        bounds=row["reference_signed_original_caps"]
        assert bounds["original_coefficient_count"]==row["length"]
        assert bounds["compressed_coefficient_count"]==(
            (row["length"]-1)//row["stride"]+1)
        assert 0<bounds["original_selected_packed_bits"]<=120_000_000
        assert 0<bounds["compressed_selected_packed_bits"]<=120_000_000
        assert bounds["original_witness_max_bits"]<=32768
        nonce_sha=row["verifier_challenge_nonce_sha256"]
        assert set(nonce_sha)==set(AUTH_ARMS)
        nonce_set=set()
        for auth_arm in AUTH_ARMS:
            assert len(nonce_sha[auth_arm])==5
            assert len(set(nonce_sha[auth_arm]))==5
            assert all(len(y)==64 for y in nonce_sha[auth_arm])
            nonce_set.update(nonce_sha[auth_arm])
            assert row["independent_prebuilt_fixture_signing_ns"][auth_arm]>=0
        assert len(nonce_set)==10
        med,samples=row["arm_medians_ms"],row["five_interleaved_samples_ms"]
        assert set(med)==set(samples)==set(ARMS)
        for arm in ARMS:
            times=samples[arm]
            assert len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
            assert math.isclose(statistics.median(times),med[arm],rel_tol=1e-12)
        ratio_map={
            "raw_gmp_over_native":med[ARMS[1]]/med[ARMS[0]],
            "signed_compact_over_native":med[ARMS[2]]/med[ARMS[0]],
            "signed_compact_over_signed_json":med[ARMS[2]]/med[ARMS[3]],
            "signed_compact_over_raw_gmp":med[ARMS[2]]/med[ARMS[1]],
            "signed_json_over_raw_gmp":med[ARMS[3]]/med[ARMS[1]],
            "compact_signed_wire_over_full_json_payload":
                row["compact_signed_wire_bytes"]/row["json_witness_payload_bytes"],
            "compact_signed_wire_over_signed_json_wire":
                row["compact_signed_wire_bytes"]/row["json_signed_wire_bytes"]}
        for k,v in ratio_map.items():
            assert math.isclose(row[k],v,rel_tol=1e-12)
    sparse_long=[r for r in rows if r["length"]==3073 and r["stride"]>1]
    sparse=[r for r in rows if r["stride"]>1]
    assert len(sparse_long)==16 and len(sparse)==48
    observed={
        "sparse_m3073_signed_compact_over_native":statistics.median(
            r["signed_compact_over_native"] for r in sparse_long),
        "sparse_m3073_signed_compact_over_signed_json":statistics.median(
            r["signed_compact_over_signed_json"] for r in sparse_long),
        "all_sparse_signed_compact_wire_over_full_json_payload":statistics.median(
            r["compact_signed_wire_over_full_json_payload"] for r in sparse)}
    for k,v in PRIMARY.items():
        assert math.isclose(observed[k],v,abs_tol=1e-12)
    assert observed["sparse_m3073_signed_compact_over_native"]<.85
    assert observed["sparse_m3073_signed_compact_over_signed_json"]>=.85
    assert observed["all_sparse_signed_compact_wire_over_full_json_payload"]<.55
    totals=x["secondary_matched_source_median_workload"]["sum_source_median_ms"]
    for arm in ARMS:
        assert math.isclose(totals[arm],
                            sum(r["arm_medians_ms"][arm] for r in rows),rel_tol=1e-12)
    assert secondary["schema"]=="nor55.first_source_matched_local_loopback_secondary.v1"
    assert secondary["original_first_CPU_sha256"]==CPU_SHA
    assert secondary["source_only_sha256"]==SOURCE_SHA
    assert secondary["source_count"]==16 and secondary["repeats"]==3
    assert secondary["producer_and_verifier_in_distinct_processes"] is True
    assert secondary["server_startup_and_nonce_issuance_excluded_from_timed_requests"]
    assert secondary["producer_inverse_codec_signature_transport_and_verified_root_included"]
    loop_rows=secondary["source_records"]
    assert len(loop_rows)==16
    seen=set()
    loop_ratios=[]
    all_time=[]
    for row in loop_rows:
        sid=row["source_sha256"]
        assert sid in public_ids and sid not in seen
        seen.add(sid)
        core=by_id[sid]
        assert core["length"]==row["full_length"]==3073
        assert core["stride"]==row["stride"]>1
        assert row["all_three_independent_verifier_responses_matched"]
        assert row["separate_producer_and_verifier_processes"]
        assert len(row["full_exact_merkle_root"])==64
        assert row["full_exact_merkle_root"]==next(
            r["reference_full_merkle_root_sha256"] for r in rows if r["source_sha256"]==sid)
        assert len(row["full_exact_coefficients_SHA256"])==64
        timings=row["producer_to_verified_response_ms"]
        native_ms=row["native_compressed_same_job_baseline_ms"]
        assert len(timings)==len(native_ms)==3
        assert all(x>0 for x in timings+native_ms)
        assert all(x>4 for x in row["request_bytes"]+row["response_bytes"])
        assert len(row["request_bytes"])==len(row["response_bytes"])==3
        assert row["server_startup_ms_outside_repeats"]>0
        ratio=statistics.median(timings)/statistics.median(native_ms)
        assert math.isclose(row["ratio_of_per_source_medians"],ratio,rel_tol=1e-12)
        loop_ratios.append(ratio)
        all_time.extend(timings)
    assert len(seen)==16
    assert math.isclose(statistics.median(loop_ratios),1.6772228552777868,abs_tol=1e-12)
    assert math.isclose(secondary["median_sourcewise_loopback_over_same_job_native"],
                        statistics.median(loop_ratios),abs_tol=1e-12)
    assert math.isclose(secondary["p95_of_48_request_response_samples_ms"],
                        sorted(all_time)[45],abs_tol=1e-12)
    return {"first_run":FIRST_RUN,"raw_SHA":CPU_SHA,
            "verdict":"FAIL (4/5)","primary":observed,
            "secondary_tcp_sourcewise_median":statistics.median(loop_ratios)}

if __name__=="__main__":
    print("NOR55 immutable original CPU/source plus secondary loopback SHA audit",audit())
