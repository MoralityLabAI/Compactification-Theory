"""NOR53 immutable first original 60-source CPU PASS read-only audit.

Reads the permanently preserved first hosted CPU/source bytes; does not rerun
source generation or CPU timing. All first registered 5/5 gates are immutable.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent/"results"
REGISTERED="ae56f414251a8485d127fc8f9899a24237870adf"
FIRST_RUN=44282552137
FIRST_CPU_ARTIFACT=14418133716
FIRST_SOURCE_ARTIFACT=14416469176
CPU_BYTES=142008
CPU_SHA="b1e4f810624568af8b53794fa9e0a4890961f7f6b5f8dbd3d4c766e511974dee"
SOURCE_BYTES=1473688
SOURCE_SHA="b7e65cdacefced6fcf1084922fdaeaaa4da5af1cf9066681471e4be1cd99ad61"
ARMS=("original_full_gmp_root","fast_full_gmp_root",
      "original_compressed_gmp_root","fast_compressed_gmp_root")
GATES={
 "all_60_fast_full_over_original_gmp_under_0p70":True,
 "longest_m3073_high_bit_fast_full_over_original_under_0p50":True,
 "all_48_sparse_fast_compressed_over_original_under_0p85":True,
 "all_exact_full_merkle_roots_and_hostile_witness_tests":True,
 "first_source_60_reentrant_caps_five_samples_original_sha_receipt":True
}
PRIMARY={
 "median_all_60_fast_full_over_original_full":0.14345133923467007,
 "median_eight_longest_high_bit_fast_full_over_original":0.07246419730786699,
 "median_48_sparse_fast_compressed_over_original":0.5968532612923032
}

def audit():
    archive=HERE/"nor53_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor53_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor53.first_original_60_source_cpu_archive.v1"
    assert receipt["original_run"]==FIRST_RUN
    assert receipt["first_cpu_artifact_id"]==FIRST_CPU_ARTIFACT
    assert receipt["first_source_only_artifact_id"]==FIRST_SOURCE_ARTIFACT
    assert receipt["first_registration_sha"]==REGISTERED
    assert receipt["raw_cpu_bytes"]==CPU_BYTES and receipt["raw_cpu_sha256"]==CPU_SHA
    assert receipt["raw_public_source_bytes"]==SOURCE_BYTES
    assert receipt["raw_public_source_sha256"]==SOURCE_SHA
    assert receipt["original_five_registered_gates"]==GATES
    assert receipt["original_first_engineering_pass"] is True
    assert archive.stat().st_size==receipt["archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor53_first_original_cpu.json","nor53_first_original_60_public_sources.json"}
        raw=z.read("nor53_first_original_cpu.json")
        public=z.read("nor53_first_original_60_public_sources.json")
    assert len(raw)==CPU_BYTES and sha256(raw).hexdigest()==CPU_SHA
    assert len(public)==SOURCE_BYTES and sha256(public).hexdigest()==SOURCE_SHA
    x,src=json.loads(raw),json.loads(public)
    assert x["schema"]=="nor53.first_60_source_native_gmp_packed_cpu.v1"
    assert src["schema"]=="nor53.first_60_public_only_native_gmp_sources.v1"
    assert x["first_registration_sha"]==src["first_registration_sha"]==REGISTERED
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0 and x["historical_exclusion_count"]==108
    assert src["source_count"]==60 and src["previous_excluded_source_identities"]==108
    assert src["public_only"] and src["inverse_witness_root_and_cpu_absent"]
    assert x["repeats"]==5 and x["four_original_timed_arms"]==list(ARMS)
    assert x["all_arms_have_full_vector_and_original_merkle_root"] is True
    assert x["original_full_and_compressed_resource_caps_enforced"] is True
    assert x["nor46_global_convolution_unchanged"] is True
    assert x["source_only_sha_frozen_upstream_of_CPU"] is True
    assert x["reference_witness_production_serialization_excluded_all_arms"] is True
    assert x["gmp_dependency_and_fast_backend"]["gmpy2_version"]=="2.3.2"
    assert x["gmp_dependency_and_fast_backend"]["unpack_little_endian"] is True
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is True
    rows=x["source_records"]
    public_ids={r["source_sha256"] for r in src["cases"]}
    assert len(rows)==len(public_ids)==60
    assert len({r["source_sha256"] for r in rows})==60
    assert set(r["source_sha256"] for r in rows)==public_ids
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert Counter(r["profile"] for r in rows)=={
        "s1-b2":12,"s4-b2":12,"s8-b2":12,"s4-b8":12,"s4-b16":12}
    assert Counter(r["family"] for r in rows)=={
        "signed_random":30,"alternating":30}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    by_id={r["source_sha256"]:r for r in src["cases"]}
    for row in rows:
        assert row["source_success"] and row["all_four_full_integer_vectors_and_merkle_roots_equal"]
        core=by_id[row["source_sha256"]]
        for key in ("replicate","length","profile","family","stride",
                    "coefficient_magnitude_bits"):
            assert row[key]==core[key]
        assert len(row["original_full_merkle_root_sha256"])==64
        assert row["full_witness_canonical_json_bytes"]>0
        assert row["reference_witness_generation_ns_outside_arms"]>=0
        assert row["reference_witness_serialization_ns_outside_arms"]>=0
        bounds=row["resource_bounds"]
        assert bounds["original_coefficient_count"]==row["length"]
        assert bounds["compressed_coefficient_count"]==(
            (row["length"]-1)//row["stride"]+1)
        assert 0<bounds["original_selected_packed_bits"]<=120000000
        assert 0<bounds["compressed_selected_packed_bits"]<=120000000
        assert bounds["original_witness_max_bits"]<=32768
        med=row["arm_medians_ms"]
        samples=row["five_interleaved_samples_ms"]
        assert set(med)==set(samples)==set(ARMS)
        for arm in ARMS:
            times=samples[arm]
            assert len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
            assert math.isclose(statistics.median(times),med[arm],rel_tol=1e-12)
        ratios={
          "fast_full_over_original_full":med[ARMS[1]]/med[ARMS[0]],
          "fast_compressed_over_original_compressed":med[ARMS[3]]/med[ARMS[2]],
          "original_compressed_over_original_full":med[ARMS[2]]/med[ARMS[0]],
          "fast_compressed_over_original_full":med[ARMS[3]]/med[ARMS[0]]
        }
        for key,value in ratios.items():
            assert math.isclose(row[key],value,rel_tol=1e-12)
    high=[r for r in rows if r["length"]==3073 and
          r["profile"] in ("s4-b8","s4-b16")]
    sparse=[r for r in rows if r["stride"]>1]
    assert len(high)==8 and len(sparse)==48
    observed={
      "median_all_60_fast_full_over_original_full":statistics.median(
          r["fast_full_over_original_full"] for r in rows),
      "median_eight_longest_high_bit_fast_full_over_original":statistics.median(
          r["fast_full_over_original_full"] for r in high),
      "median_48_sparse_fast_compressed_over_original":statistics.median(
          r["fast_compressed_over_original_compressed"] for r in sparse)
    }
    assert receipt["original_primary_results"]==PRIMARY
    for key,val in PRIMARY.items():
        assert math.isclose(observed[key],val,abs_tol=1e-12)
    assert observed["median_all_60_fast_full_over_original_full"]<.70
    assert observed["median_eight_longest_high_bit_fast_full_over_original"]<.50
    assert observed["median_48_sparse_fast_compressed_over_original"]<.85
    totals=x["secondary_ten_block_workload"]["source_median_total_ms"]
    for a in ARMS:
        assert math.isclose(totals[a],sum(r["arm_medians_ms"][a] for r in rows),
                            rel_tol=1e-12)
    return {"first_run":FIRST_RUN,"cpu_sha":CPU_SHA,
            "verdict":"PASS (5/5)","source_units":60,"primary":observed}

if __name__=="__main__":
    print("NOR53 original immutable evidence audit",audit())
