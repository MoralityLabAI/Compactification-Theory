"""NOR52 permanent immutable first 60-source original CPU PASS audit.

Reads byte-identical first raw CPU/source files inside source-controlled ZIP.
Does not generate any first source, recompute an inverse or rerun timings.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent/"results"
REGISTRATION="9526627e21b429e7fea1928b3ddfb0e1dadc5e74"
FIRST_RUN=44275345207
FIRST_CPU_ARTIFACT=14413918829
FIRST_SOURCE_ARTIFACT=14414901378
CPU_BYTES=166277
CPU_SHA="83090c4d9647b543b8950025808f087676ffd8a6dee03709b729bd663304a99d"
SOURCE_BYTES=1474292
SOURCE_SHA="d18b795053d81c16fc5316b490316e90fe6b373156a8ee8b295008a13df150f4"
ARMS=("native_full_R_root","native_compressed_R_root",
      "gmp_full_verify_root","gmp_compressed_verify_root")
PHASES=("parameters_ns","packing_ns","multiplication_ns",
        "extraction_ns","correction_ns")
GATES={
 "sparse_m3073_native_compressed_over_full_under_0p50":True,
 "sparse_m3073_gmp_compressed_over_full_under_0p50":True,
 "dense_s1_both_compressed_over_full_under_1p10":True,
 "all_exact_output_roots_and_source_free_adversarial_controls":True,
 "frozen_60_source_five_interleaved_reentrant_caps_and_phase_parity":True
}

def audit():
    archive=HERE/"nor52_first_original_hosted.zip"
    receipt=json.loads((HERE/"nor52_first_receipt.json").read_bytes())
    assert receipt["schema"]=="nor52.first_original_60_source_cpu_archive.v1"
    assert receipt["first_run"]==FIRST_RUN
    assert receipt["first_cpu_artifact_id"]==FIRST_CPU_ARTIFACT
    assert receipt["first_source_only_artifact_id"]==FIRST_SOURCE_ARTIFACT
    assert receipt["first_registration_sha"]==REGISTRATION
    assert receipt["raw_cpu_bytes"]==CPU_BYTES and receipt["raw_cpu_sha256"]==CPU_SHA
    assert receipt["raw_source_bytes"]==SOURCE_BYTES and receipt["raw_source_sha256"]==SOURCE_SHA
    assert receipt["first_original_five_gates"]==GATES
    assert receipt["first_original_engineering_pass"] is True
    assert archive.stat().st_size==receipt["archive_zip_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["archive_zip_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor52_first_original_cpu.json","nor52_first_public_only_60_sources.json"}
        cpu=z.read("nor52_first_original_cpu.json")
        public=z.read("nor52_first_public_only_60_sources.json")
    assert len(cpu)==CPU_BYTES and sha256(cpu).hexdigest()==CPU_SHA
    assert len(public)==SOURCE_BYTES and sha256(public).hexdigest()==SOURCE_SHA
    x,fixture=json.loads(cpu),json.loads(public)
    assert x["schema"]=="nor52.first_60_exact_sparse_support_compression_cpu.v1"
    assert fixture["schema"]=="nor52.first_public_only_60_sparse_dense_sources.v1"
    assert x["registration_first_sha"]==fixture["first_registration_sha"]==REGISTRATION
    assert x["first_public_only_source_sha256"]==SOURCE_SHA
    assert x["source_count"]==x["source_success_count"]==60
    assert x["source_failure_count"]==0
    assert fixture["source_count"]==60 and fixture["public_only"]
    assert fixture["no_inverse_trace_root_or_timing"]
    assert x["repeats"]==5 and x["arms"]==list(ARMS)
    assert x["full_merkle_root_in_all_arm_timings"] is True
    assert x["reference_production_serialization_outside_timings"] is True
    assert x["full_witness_original_resource_caps_enforced"] is True
    assert x["public_only_sha_frozen_upstream_of_timing"] is True
    assert x["no_NOR46_global_override"] is True
    assert x["gmp_dependency"]["gmpy2_version"]=="2.3.2"
    assert x["first_original_five_registered_gates"]==GATES
    assert x["first_original_engineering_pass"] is True
    cases=fixture["cases"]
    public_ids={r["source_sha256"] for r in cases}
    assert len(cases)==len(public_ids)==60
    rows=x["source_records"]
    ids=[r["source_sha256"] for r in rows]
    assert len(rows)==len(set(ids))==60 and set(ids)==public_ids
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert Counter(r["profile"] for r in rows)=={
         "s1-b2":12,"s4-b2":12,"s8-b2":12,"s4-b8":12,"s4-b16":12}
    assert Counter(r["family"] for r in rows)=={"signed_random":30,"alternating":30}
    assert Counter(r["replicate"] for r in rows)=={0:30,1:30}
    cores={r["source_sha256"]:r for r in cases}
    for row in rows:
        assert row["source_success"] and row["full_exact_inverse_vector_and_merkle_root_equal"]
        core=cores[row["source_sha256"]]
        for k in ("replicate","length","profile","family","stride",
                  "coefficient_magnitude_bits"):
            assert row[k]==core[k]
        assert len(row["reference_merkle_root_sha256"])==64
        assert row["full_inverse_canonical_json_bytes"]>0
        assert row["reference_inverse_generation_ns_outside_arms"]>=0
        assert row["witness_serialization_ns_outside_arms"]>=0
        bounds=row["original_and_compressed_resource_bounds"]
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
            "native_compressed_over_full":med[ARMS[1]]/med[ARMS[0]],
            "gmp_compressed_over_full":med[ARMS[3]]/med[ARMS[2]],
            "gmp_full_over_native_full":med[ARMS[2]]/med[ARMS[0]],
            "gmp_compressed_over_native_compressed":med[ARMS[3]]/med[ARMS[1]]}
        for name,ratio in ratios.items():
            assert math.isclose(row[name],ratio,rel_tol=1e-12)
        phase=row["untimed_original_full_gmp_phases"]
        assert phase["exact_signed_identity"] and phase["reference_kernel_equal"]
        assert phase["full_packed_bits"]==bounds["original_selected_packed_bits"]
        assert all(type(phase[p]) is int and phase[p]>=0 for p in PHASES)
        assert phase["phase_total_ns"]==sum(phase[p] for p in PHASES)
    large=[r for r in rows if r["length"]==3073 and r["stride"]>1]
    dense=[r for r in rows if r["stride"]==1]
    assert len(large)==16 and len(dense)==12
    expected={
       "sparse_native":0.16342142553791797,
       "sparse_gmp":0.07866323063716676,
       "dense_native":1.0011657433543402,
       "dense_gmp":1.003117652709324
    }
    measured={
       "sparse_native":statistics.median(r["native_compressed_over_full"] for r in large),
       "sparse_gmp":statistics.median(r["gmp_compressed_over_full"] for r in large),
       "dense_native":statistics.median(r["native_compressed_over_full"] for r in dense),
       "dense_gmp":statistics.median(r["gmp_compressed_over_full"] for r in dense)
    }
    for key in expected:
        assert math.isclose(measured[key],expected[key],abs_tol=1e-12)
    assert measured["sparse_native"]<.5 and measured["sparse_gmp"]<.5
    assert measured["dense_native"]<1.1 and measured["dense_gmp"]<1.1
    assert receipt["original_primary_results"]==measured
    totals=x["secondary_source_median_workload"]["source_median_total_ms"]
    for arm in ARMS:
        assert math.isclose(totals[arm],sum(r["arm_medians_ms"][arm] for r in rows),
                            rel_tol=1e-12)
    return {"first_run":FIRST_RUN,"original_sha":CPU_SHA,"verdict":"PASS (5/5)",
            "primary_sourcewise":measured,"source_count":60}

if __name__=="__main__":
    print("NOR52 first immutable original evidence PASS",audit())
