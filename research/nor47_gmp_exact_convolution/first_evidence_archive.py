"""NOR-47 first original historical CPU evidence archive.

Read-only input. The file's original measured timings, including negative
engineering outcomes, are copied byte-for-byte. No source or benchmark is
rerun or adjusted by this archiver.
"""
from __future__ import annotations
import argparse,json,math,statistics
from hashlib import sha256
from pathlib import Path

FIRST_RUN=44240888113
FIRST_ARTIFACT_ID=14400450017
FIRST_RAW_BYTES=34928
FIRST_RAW_SHA="809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530"
SOURCE_SHA="d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052"
CPU_SHA="4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64"
EXPECTED_REPS=(100,104,108,112)


def archive(input_path:Path,output_dir:Path):
    first=input_path.read_bytes()
    assert len(first)==FIRST_RAW_BYTES
    assert sha256(first).hexdigest()==FIRST_RAW_SHA
    grade=json.loads(first)
    assert grade["schema"]=="nor47.first_archived_gmp_exact_convolution_cpu.v1"
    assert grade["design_issue"]==248
    assert grade["nor46_original_first_trace_sha256"]==SOURCE_SHA
    assert grade["nor46_original_first_cpu_sha256"]==CPU_SHA
    assert grade["source_selection"]==list(EXPECTED_REPS)
    assert grade["historical_source_pairs"]==16
    assert grade["new_prospective_nor47_source_ids"]==0
    assert grade["fixed_repeat_count_per_arm"]==5
    assert grade["gmp_backend"]["gmpy2_version"]=="2.3.2"
    assert grade["exact_integer_arithmetic_without_modular_reduction"]
    assert grade["all_signature_program_source_and_root_semantics_unchanged"]
    assert grade["full_merkle_root_computation_included_in_all_verifier_arms"]
    assert grade["shared_signature_session_transport_excluded_in_all_timings"]
    assert grade["all_first_source_parity"]
    assert grade["randomized_proof_soundness_claim"] is False
    assert grade["claimed_succinctness"] is False
    records=grade["source_records"]
    assert len(records)==len({r["source_sha256"] for r in records})==16
    assert {r["replicate"] for r in records}==set(EXPECTED_REPS)
    assert sum(r["degree"]==512 for r in records)==8
    assert sum(r["degree"]==1024 for r in records)==8
    assert sum(r["has_reciprocal_R"] for r in records)==12
    arms={
        "nor45_direct","nor46_python_packed",
        "nor47_gmp_packed","nor47_python_pack_gmp_multiply"
    }
    for r in records:
        assert r["all_signed_exact_integer_and_root_parity"]
        assert set(r["randomized_interleaved_arm_samples_ms"])==arms
        assert all(len(times)==5 and all(math.isfinite(t) and t>0 for t in times)
                   for times in r["randomized_interleaved_arm_samples_ms"].values())
        a=r["randomized_interleaved_arm_samples_ms"]
        med={arm:statistics.median(x) for arm,x in a.items()}
        assert math.isclose(med["nor47_gmp_packed"],
                            r["nor47_gmp_median_ms"],abs_tol=1e-12)
        assert math.isclose(med["nor45_direct"],
                            r["nor45_original_median_ms"],abs_tol=1e-12)
        assert math.isclose(
            med["nor47_gmp_packed"]/med["nor45_direct"],
            r["ratios"]["gmp_over_nor45"],rel_tol=1e-12)
    means=grade["summary"]
    assert math.isclose(
        statistics.median(r["ratios"]["gmp_over_nor45"] for r in records),
        means["median_per_source_gmp_over_nor45"],rel_tol=1e-12)
    assert math.isclose(
        statistics.median(r["ratios"]["gmp_over_nor46"] for r in records),
        means["median_per_source_gmp_over_nor46"],rel_tol=1e-12)
    assert grade["first_engineering_targets"]=={
        "gmp_faster_than_nor45":
            means["median_per_source_gmp_over_nor45"]<1,
        "gmp_at_least_twofold_faster_than_python_trace":
            means["median_per_source_gmp_over_nor46"]<.5}
    assert grade["first_engineering_pass"]==all(
        grade["first_engineering_targets"].values())
    output_dir.mkdir(parents=True,exist_ok=True)
    rawpath=output_dir/"nor47_first_original_cpu.json"
    assert not rawpath.exists()
    rawpath.write_bytes(first)
    receipt={
        "schema":"nor47.original_first_source_locked_cpu_archive.v1",
        "first_original_run":FIRST_RUN,
        "first_original_artifact_id":FIRST_ARTIFACT_ID,
        "original_source_units":16,
        "prospective_new_source_units":0,
        "original_nor46_trace_sha256":SOURCE_SHA,
        "original_nor46_cpu_sha256":CPU_SHA,
        "original_first_raw_cpu_bytes":len(first),
        "original_first_raw_cpu_sha256":sha256(first).hexdigest(),
        "original_first_hardware":grade["cpu_model"],
        "gmp_version":grade["gmp_backend"],
        "five_repeats_per_verifier_arm":5,
        "original_engineering_targets":grade["first_engineering_targets"],
        "original_engineering_pass":grade["first_engineering_pass"],
        "original_summary":means,
        "limitation":"Exact signed integer historical-only benchmarking; "
                     "no cryptographic proof-system soundness or true program authority."
    }
    (output_dir/"nor47_first_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("NOR47 ORIGINAL FIRST RAW CPU BYTES/SHA",
          len(first),sha256(first).hexdigest())
    print("NOR47 FIRST ENGINEERING PASS",grade["first_engineering_pass"])
    print("NOR47 FIRST ORIGINAL SPEED RATIOS",means)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    archive(args.original,args.output_dir)


if __name__=="__main__":
    main()
