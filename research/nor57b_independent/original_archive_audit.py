"""NOR57B first original full signed selector CPU/source immutable SHA audit.

Reads two exact first uploaded raw JSON byte streams, original source
identities, every five-repeat source median and frozen 4-of-5 negative score.
Never regenerates first sources or retimes full exact signed witnesses.
NOR57's earlier original unscored reporting FAIL remains separate.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent/"results"
REGISTRATION="0a9467ae18df3296d8fdf04971f0ffbbb4c6e3d7"
FIRST_RUN=44305694146
FIRST_SOURCE_ARTIFACT=14428291671
FIRST_CPU_ARTIFACT=14427713264
SOURCE_BYTES=1474323
SOURCE_SHA="bca8cddca58a4c8a234df4f6bba2b31c44c7e114c2f07f67a545638dad5b9a63"
CPU_BYTES=173922
CPU_SHA="83fbda5e6b94c3e434203def01f3a2092235a73dc3e235f7d2187450dc421890"
ARMS=("always_fast_GMP_signed_v2","always_native_recompute_signed_v2",
      "frozen_public_selector_signed_v2")
GATES={
 "all_60_selector_sum_source_median_over_GMP_under_0p95":False,
 "eight_native_selected_native_over_GMP_median_under_0p85":True,
 "fifty_two_GMP_selected_selector_over_GMP_median_under_1p10":True,
 "all_60_signed_exact_full_vectors_original_roots_and_tamper_controls":True,
 "first_source_60_348_old_excluded_5_repeats_fixed_route_caps_archive":True
}
PRIMARY={
 "whole_source_median_selector_over_GMP":0.9899612598104661,
 "eight_native_selected_native_over_GMP":0.7383314447266901,
 "fifty_two_GMP_selected_selector_over_GMP":1.021210225903338
}

def read_original():
    archive=HERE/"nor57b_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor57b_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor57b.first_original_signed_source_selector_CPU_archive.v1"
    assert receipt["first_run"]==FIRST_RUN
    assert receipt["original_source_artifact_id"]==FIRST_SOURCE_ARTIFACT
    assert receipt["original_CPU_artifact_id"]==FIRST_CPU_ARTIFACT
    assert receipt["registration_sha_before_new_source"]==REGISTRATION
    assert receipt["raw_source_bytes"]==SOURCE_BYTES
    assert receipt["raw_source_sha256"]==SOURCE_SHA
    assert receipt["raw_CPU_bytes"]==CPU_BYTES
    assert receipt["raw_CPU_sha256"]==CPU_SHA
    assert receipt["original_five_registered_gates"]==GATES
    assert receipt["first_original_engineering_pass"] is False
    assert receipt["original_primary"]==PRIMARY
    assert archive.stat().st_size==receipt["archive_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["archive_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor57b_first_original_cpu.json",
            "nor57b_first_original_60_public_only.json"
        }
        raw=z.read("nor57b_first_original_cpu.json")
        public=z.read("nor57b_first_original_60_public_only.json")
    assert len(raw)==CPU_BYTES and sha256(raw).hexdigest()==CPU_SHA
    assert len(public)==SOURCE_BYTES and sha256(public).hexdigest()==SOURCE_SHA
    x,fixture=json.loads(raw),json.loads(public)
    assert x["schema"]=="nor57b.first_signed_v2_public_source_selector_cpu.v1"
    assert fixture["schema"]=="nor57b.first_public_only_60_authenticated_router_sources.v1"
    assert x["first_registration_sha"]==fixture["first_registration_sha"]==REGISTRATION
    assert fixture["source_domain"]=="NOR57B/SOURCE/v1"
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert x["previous_original_source_exclusions"]==348  # frozen stale descriptive field
    assert fixture["source_count"]==60 and fixture["excluded_original_source_IDs"]==408
    assert receipt["source_exclusions_metadata_discrepancy"]=={"original_CPU_metadata_reports":348,"upstream_public_roster_and_source_check_excluded":408}
    assert fixture["public_only"] and fixture["no_inverse_root_payload_signature_session_or_timing"]
    assert x["repeat_count"]==5 and x["arms"]==list(ARMS)
    assert x["source_first_SHA_frozen_before_witness_and_timing"]
    assert x["exact_same_signed_v2_wire_for_all_three_arms"]
    assert x["same_common_signature_decode_full_caps_and_replay_pipeline"]
    assert x["full_original_witness_and_merkle_root_each_arm"]
    assert x["witness_and_fixture_signing_outside_timed_verifier"]
    assert x["original_reference_NOR56_signed_v2_parity_checked_before_timing"]
    assert x["selector_ID"]=="NOR57-FROZEN-PUBLIC-SELECTOR-v1"
    assert x["registered_selector_rule"]=={
        "native_if_full_length_is_one_of":[769,1537],
        "native_if_trusted_stride_equals":4,
        "native_if_actual_public_max_coefficient_bit_length_equals":16,
        "otherwise":"GMP","never_reads_untrusted_witness_or_session_nonce":True
    }
    assert x["selector_native_count"]==8 and x["selector_gmp_count"]==52
    assert x["nor46_global_exact_convolution_unchanged"]
    assert x["gmp_backend"]["gmpy2_version"]=="2.3.2"
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
    indexed={r["source_sha256"]:r for r in cases}
    native=[]
    gmp=[]
    for r in rows:
        assert r["source_success"] and r["exact_full_signed_vector_and_original_merkle_all_three"]
        c=indexed[r["source_sha256"]]
        for key in ("replicate","length","profile","family","stride",
                    "coefficient_magnitude_bits"):
            assert r[key]==c[key]
        assert len(r["original_full_merkle_root_sha256"])==64
        assert r["full_native_reference_generation_ns_outside_arms"]>=0
        assert r["full_reference_json_witness_bytes"]>0
        assert r["full_reference_json_serialization_ns_outside_arms"]>=0
        assert r["prebuilt_fixture_signer_encode_ns"]>=0
        assert r["same_signed_envelope_bytes"]>r["reduced_signed_payload_bytes"]>0
        nonces=r["five_distinct_verifier_challenge_nonce_SHA256"]
        assert len(nonces)==len(set(nonces))==5
        assert all(len(h)==64 for h in nonces)
        f=tuple(int(z) for z in c["source_coefficients_decimal"])
        expected_width=max(abs(z).bit_length() for z in f)
        assert r["trusted_public_max_coefficient_bitwidth"]==expected_width
        expected=("NATIVE" if c["stride"]==4 and
                  c["length"] in (769,1537) and expected_width==16 else "GMP")
        assert r["frozen_public_selector_decision"]==expected
        (native if expected=="NATIVE" else gmp).append(r)
        bounds=r["full_and_compressed_original_resource_bounds"]
        assert bounds["original_coefficient_count"]==r["length"]
        assert bounds["compressed_coefficient_count"]==(
            (r["length"]-1)//r["stride"]+1)
        assert 0<bounds["original_selected_packed_bits"]<=120_000_000
        assert 0<bounds["compressed_selected_packed_bits"]<=120_000_000
        assert bounds["original_witness_max_bits"]<=32768
        med,samples=r["arm_medians_ms"],r["five_interleaved_samples_ms"]
        assert set(med)==set(samples)==set(ARMS)
        for arm in ARMS:
            times=samples[arm]
            assert len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
            assert math.isclose(statistics.median(times),med[arm],rel_tol=1e-12)
        checks={
            "always_native_over_always_GMP":med[ARMS[1]]/med[ARMS[0]],
            "frozen_selector_over_always_GMP":med[ARMS[2]]/med[ARMS[0]],
            "frozen_selector_over_always_native":med[ARMS[2]]/med[ARMS[1]]
        }
        for k,value in checks.items():
            assert math.isclose(r[k],value,rel_tol=1e-12)
    assert len(native)==8 and len(gmp)==52
    totals={a:sum(r["arm_medians_ms"][a] for r in rows) for a in ARMS}
    primary={
        "whole_source_median_selector_over_GMP":totals[ARMS[2]]/totals[ARMS[0]],
        "eight_native_selected_native_over_GMP":statistics.median(
            r["always_native_over_always_GMP"] for r in native),
        "fifty_two_GMP_selected_selector_over_GMP":statistics.median(
            r["frozen_selector_over_always_GMP"] for r in gmp)
    }
    assert primary==PRIMARY
    assert primary["whole_source_median_selector_over_GMP"]>=.95
    assert primary["eight_native_selected_native_over_GMP"]<.85
    assert primary["fifty_two_GMP_selected_selector_over_GMP"]<1.1
    assert x["primary_selector_ratio_of_sum_of_sixty_source_medians"]==PRIMARY[
        "whole_source_median_selector_over_GMP"]
    assert totals==x["secondary_ten_block_workload"]["sum_of_source_median_ms"]
    assert receipt["original_workload_ms"]==totals
    return {"first_run":FIRST_RUN,"original_CPU_SHA":CPU_SHA,
            "first_verdict":"FAIL (4/5)","source_count":60,
            "native_cases":8,"GMP_cases":52,"primary":primary}

if __name__=="__main__":
    print("NOR57B immutable first source/CPU and five gate audit",read_original())
