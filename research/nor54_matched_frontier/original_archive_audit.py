"""NOR54 independent read-only archive audit of original first engineering FAIL.

Verifies first CPU and source SHA, registered score and all five arm samples,
without generating new sources or rerunning any inverse/CPU measurements.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent/"results"
REGISTERED="10e7ac3c3fdefec89d90d7a98a947099db8af86e"
FIRST_RUN=44288506867
CPU_ARTIFACT=14420281905
SOURCE_ARTIFACT=14420037504
CPU_BYTES=177615
CPU_SHA="27f7926a53aee3bd25520a1c9e227041997b4bdb9a92a8876cd73af267f67ed1"
SOURCE_BYTES=1474198
SOURCE_SHA="9ac35fdee19a8b9a5b2d08c1ff890bddb3141b69232c4a2687d46195c37a627d"
ARMS=("full_native_R_root","compressed_native_R_root",
      "fast_full_gmp_witness_root","fast_compressed_gmp_witness_root",
      "compressed_produce_serialize_fast_verify_root")
GATES={
 "sparse_m3073_fast_compressed_gmp_over_compressed_native_under_0p85":True,
 "sparse_m3073_local_produce_serialize_verify_over_native_under_1p50":False,
 "dense_s1_native_and_gmp_compressed_over_full_under_1p10":True,
 "all_60_exact_full_vectors_merkle_and_hostile_source_controls":True,
 "first_60_source_SHA_five_repeats_caps_and_reentrant_archive":True
}
PRIMARY={
 "sparse_verifier_over_compressed_native":0.7400299551611904,
 "sparse_local_produce_serialize_verify_over_compressed_native":1.7117850978796483,
 "dense_native_compressed_over_full":1.0052026991161207,
 "dense_fast_gmp_compressed_over_full":1.0146430592944196
}

def audit():
    archive=HERE/"nor54_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor54_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor54.first_original_60_source_CPU_archive.v1"
    assert receipt["original_run"]==FIRST_RUN
    assert receipt["first_cpu_artifact_id"]==CPU_ARTIFACT
    assert receipt["first_public_source_artifact_id"]==SOURCE_ARTIFACT
    assert receipt["registration_first_sha"]==REGISTERED
    assert receipt["original_CPU_bytes"]==CPU_BYTES
    assert receipt["original_CPU_sha256"]==CPU_SHA
    assert receipt["original_source_bytes"]==SOURCE_BYTES
    assert receipt["original_source_sha256"]==SOURCE_SHA
    assert receipt["original_first_five_registered_gates"]==GATES
    assert receipt["first_original_engineering_pass"] is False
    assert receipt["original_primary"]==PRIMARY
    assert archive.stat().st_size==receipt["original_archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["original_archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor54_first_original_cpu.json","nor54_first_original_60_public_only.json"}
        cpu=z.read("nor54_first_original_cpu.json")
        source=z.read("nor54_first_original_60_public_only.json")
    assert len(cpu)==CPU_BYTES and sha256(cpu).hexdigest()==CPU_SHA
    assert len(source)==SOURCE_BYTES and sha256(source).hexdigest()==SOURCE_SHA
    result,fixture=json.loads(cpu),json.loads(source)
    assert result["schema"]=="nor54.first_60_matched_native_fast_gmp_e2e_cpu.v1"
    assert fixture["schema"]=="nor54.first_public_only_60_matched_exact_series.v1"
    assert result["first_registration_sha"]==fixture["first_registration_sha"]==REGISTERED
    assert result["first_public_only_source_sha256"]==SOURCE_SHA
    assert fixture["source_domain"]=="NOR54/SOURCE/v1"
    assert result["source_count"]==result["source_success_count"]==60
    assert result["source_failure_count"]==0 and result["previous_excluded_source_count"]==168
    assert fixture["source_count"]==60 and fixture["original_excluded_source_count"]==168
    assert fixture["public_only"] and fixture["no_inverse_witness_root_or_timing"]
    assert result["repeats"]==5 and result["five_original_timed_arms"]==list(ARMS)
    assert result["all_arms_have_complete_exact_output_and_original_full_root"] is True
    assert result["local_e2e_charges_production_serialization_parse_validation"] is True
    assert result["original_full_caps_enforced"] and result["nor46_process_global_unchanged"]
    assert result["public_only_sha_frozen_upstream_of_CPU"]
    assert result["gmp_backend"]["gmpy2_version"]=="2.3.2"
    assert result["first_original_five_registered_gates"]==GATES
    assert result["first_original_engineering_pass"] is False
    rows=result["source_records"]
    cases=fixture["cases"]
    source_ids={x["source_sha256"] for x in cases}
    assert len(source_ids)==60
    assert len(rows)==60 and {x["source_sha256"] for x in rows}==source_ids
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert Counter(r["profile"] for r in rows)=={
        "s1-b2":12,"s4-b2":12,"s8-b2":12,"s4-b8":12,"s4-b16":12}
    assert Counter(r["family"] for r in rows)=={
        "signed_random":30,"alternating":30}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    by_id={x["source_sha256"]:x for x in cases}
    for row in rows:
        assert row["source_success"] and row["all_five_full_exact_vectors_and_merkle_roots_equal"]
        core=by_id[row["source_sha256"]]
        for key in ("replicate","length","profile","family","stride",
                    "coefficient_magnitude_bits"):
            assert row[key]==core[key]
        assert len(row["reference_full_merkle_root_sha256"])==64
        assert row["independent_full_witness_json_bytes"]>0
        assert row["reference_full_native_generation_ns_outside_arms"]>=0
        assert row["independent_full_witness_json_serialization_ns_outside_arms"]>=0
        bounds=row["full_and_compressed_signed_resource_bounds"]
        assert bounds["original_coefficient_count"]==row["length"]
        assert bounds["compressed_coefficient_count"]==(
            (row["length"]-1)//row["stride"]+1)
        assert 0<bounds["original_selected_packed_bits"]<=120000000
        assert 0<bounds["compressed_selected_packed_bits"]<=120000000
        assert bounds["original_witness_max_bits"]<=32768
        samples,medians=row["five_interleaved_samples_ms"],row["arm_medians_ms"]
        assert set(samples)==set(medians)==set(ARMS)
        for arm in ARMS:
            durations=samples[arm]
            assert len(durations)==5
            assert all(math.isfinite(t) and t>0 for t in durations)
            assert math.isclose(statistics.median(durations),medians[arm],rel_tol=1e-12)
        ratios={
            "compressed_native_over_full_native":medians[ARMS[1]]/medians[ARMS[0]],
            "fast_compressed_verify_over_compressed_native":medians[ARMS[3]]/medians[ARMS[1]],
            "fast_full_verify_over_full_native":medians[ARMS[2]]/medians[ARMS[0]],
            "fast_compressed_over_fast_full":medians[ARMS[3]]/medians[ARMS[2]],
            "produce_serialize_fast_verify_over_compressed_native":medians[ARMS[4]]/medians[ARMS[1]],
            "end_to_end_over_fast_compressed_verify":medians[ARMS[4]]/medians[ARMS[3]]
        }
        for key,val in ratios.items():
            assert math.isclose(val,row[key],rel_tol=1e-12)
    sparse=[r for r in rows if r["stride"]>1 and r["length"]==3073]
    dense=[r for r in rows if r["stride"]==1]
    assert len(sparse)==16 and len(dense)==12
    observed={
        "sparse_verifier_over_compressed_native":statistics.median(
            r["fast_compressed_verify_over_compressed_native"] for r in sparse),
        "sparse_local_produce_serialize_verify_over_compressed_native":statistics.median(
            r["produce_serialize_fast_verify_over_compressed_native"] for r in sparse),
        "dense_native_compressed_over_full":statistics.median(
            r["compressed_native_over_full_native"] for r in dense),
        "dense_fast_gmp_compressed_over_full":statistics.median(
            r["fast_compressed_over_fast_full"] for r in dense)
    }
    for key,val in PRIMARY.items():
        assert math.isclose(observed[key],val,abs_tol=1e-12)
    assert observed["sparse_verifier_over_compressed_native"]<.85
    assert observed["sparse_local_produce_serialize_verify_over_compressed_native"]>=1.50
    assert observed["dense_native_compressed_over_full"]<1.10
    assert observed["dense_fast_gmp_compressed_over_full"]<1.10
    totals=result["secondary_ten_block_source_median_workload"]["source_median_total_ms"]
    for arm in ARMS:
        assert math.isclose(totals[arm],
                            sum(r["arm_medians_ms"][arm] for r in rows),
                            rel_tol=1e-12)
    return {"run":FIRST_RUN,"CPU_SHA":CPU_SHA,"verdict":"FAIL (4/5)",
            "sources":60,"registered_primary":observed}

if __name__=="__main__":
    print("NOR54 first original negative evidence read-only audit",audit())
