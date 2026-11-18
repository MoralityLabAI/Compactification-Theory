"""NOR-47 immutable FIRST historical CPU benchmark byte and score audit.

No GMP installation required for the SHA check. Never regenerates sources,
rebenchmarks wall times, or changes the original fixed performance targets.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/"results/nor47_first_original_cpu.json"
MANIFEST=HERE/"results/nor47_first_receipt.json"
FIRST_RAW_BYTES=34928
FIRST_RAW_SHA="809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530"
FIRST_RUN=44240888113
FIRST_ARTIFACT=14400450017
NOR46_SOURCE_SHA="d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052"
NOR46_CPU_SHA="4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64"


def audit():
    raw=ORIGINAL.read_bytes()
    assert len(raw)==FIRST_RAW_BYTES
    assert sha256(raw).hexdigest()==FIRST_RAW_SHA
    source=json.loads(raw)
    manifest=json.loads(MANIFEST.read_bytes())
    assert manifest["schema"]=="nor47.original_first_source_locked_cpu_archive.v1"
    assert manifest["first_original_run"]==FIRST_RUN
    assert manifest["first_original_artifact_id"]==FIRST_ARTIFACT
    assert manifest["original_first_raw_cpu_bytes"]==FIRST_RAW_BYTES
    assert manifest["original_first_raw_cpu_sha256"]==FIRST_RAW_SHA
    assert manifest["original_engineering_pass"] is True
    assert source["schema"]=="nor47.first_archived_gmp_exact_convolution_cpu.v1"
    assert source["historical_source_pairs"]==16
    assert source["new_prospective_nor47_source_ids"]==0
    assert source["source_selection"]==[100,104,108,112]
    assert source["fixed_repeat_count_per_arm"]==5
    assert source["nor46_original_first_trace_sha256"]==NOR46_SOURCE_SHA
    assert source["nor46_original_first_cpu_sha256"]==NOR46_CPU_SHA
    assert source["first_engineering_pass"] is True
    assert source["first_engineering_targets"]=={
        "gmp_faster_than_nor45":True,
        "gmp_at_least_twofold_faster_than_python_trace":True}
    assert manifest["original_engineering_targets"]==source["first_engineering_targets"]
    assert manifest["original_summary"]==source["summary"]
    assert source["gmp_backend"]["gmpy2_version"]=="2.3.2"
    assert source["gmp_backend"]["gmp_version"]=="GMP 6.3.0"
    assert source["exact_integer_arithmetic_without_modular_reduction"]
    assert source["all_signature_program_source_and_root_semantics_unchanged"]
    assert source["full_merkle_root_computation_included_in_all_verifier_arms"]
    assert source["shared_signature_session_transport_excluded_in_all_timings"]
    assert source["all_first_source_parity"]
    records=source["source_records"]
    assert len(records)==len({r["source_sha256"] for r in records})==16
    assert Counter(r["degree"] for r in records)=={512:8,1024:8}
    assert Counter(r["has_reciprocal_R"] for r in records)=={True:12,False:4}
    assert Counter(r["replicate"] for r in records)=={
        100:4,104:4,108:4,112:4}
    assert Counter(r["family"] for r in records)=={
        "wide_dense_small":8,"wide_regime_blocks":8}
    arms={
        "nor45_direct","nor46_python_packed",
        "nor47_gmp_packed","nor47_python_pack_gmp_multiply"}
    for r in records:
        assert r["all_signed_exact_integer_and_root_parity"]
        assert set(r["randomized_interleaved_arm_samples_ms"])==arms
        samples=r["randomized_interleaved_arm_samples_ms"]
        assert all(len(v)==5 and all(math.isfinite(x) and x>0 for x in v)
                   for v in samples.values())
        assert math.isclose(
            statistics.median(samples["nor47_gmp_packed"]),
            r["nor47_gmp_median_ms"],abs_tol=1e-12)
        assert math.isclose(
            r["nor47_gmp_median_ms"]/r["nor45_original_median_ms"],
            r["ratios"]["gmp_over_nor45"],rel_tol=1e-12)
    summary=source["summary"]
    assert summary["n"]==16
    for key,source_key in [
        ("median_per_source_gmp_over_nor45","gmp_over_nor45"),
        ("median_per_source_gmp_over_nor46","gmp_over_nor46"),
        ("median_per_source_hybrid_over_nor45","hybrid_over_nor45")]:
        expected=statistics.median(r["ratios"][source_key] for r in records)
        assert math.isclose(summary[key],expected,rel_tol=1e-12)
    assert summary["median_per_source_gmp_over_nor45"]<1
    assert summary["median_per_source_gmp_over_nor46"]<.5
    assert math.isclose(summary["median_per_source_gmp_over_nor45"],
                        .8104069120627158,abs_tol=1e-12)
    assert math.isclose(summary["median_per_source_gmp_over_nor46"],
                        .12669814677924263,abs_tol=1e-12)
    return {
        "schema":"nor47.original_immutable_cpu_audit.v1",
        "first_run":FIRST_RUN,
        "first_source_locked_cpu_sha256":FIRST_RAW_SHA,
        "original_first_bytes":FIRST_RAW_BYTES,
        "original_engineering_pass":True,
        "fixed_historical_source_pairs":16,
        "no_new_prospective_sources":True,
        "same_runner_gmp_over_nor45_median_source_ratio":
            summary["median_per_source_gmp_over_nor45"],
        "same_runner_gmp_over_python_median_source_ratio":
            summary["median_per_source_gmp_over_nor46"],
        "independent_root_trace_parity":True,
    }


if __name__=="__main__":
    print(json.dumps(audit(),sort_keys=True,indent=2))
