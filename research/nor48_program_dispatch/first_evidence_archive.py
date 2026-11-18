"""NOR-48 immutable original first 16-source CPU engineering FAIL archiver.

Reads hosted first-run raw bytes only, never retimes or changes measured
parameters. The predeclared <0.75 median dispatch/direct threshold stays
FAIL when the first hosted ratio is 0.8182870781348079.
"""
from __future__ import annotations
import argparse,json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path

FIRST_RUN=44253553267
FIRST_ARTIFACT_ID=14404902340
RAW_BYTES=27609
RAW_SHA="d5cf4b0e3860907eb58c2a37b84ff631f023caf2b69e6e0d0e5a322e2f0b2b99"
PRIOR_SHA="809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530"
ARMS=("nor45_direct","nor46_python_full_trace","nor47_gmp_full_trace",
      "nor48_r_aware_dispatch")
GATES={
    "dispatcher_median_ratio_under_0p75":False,
    "all_four_no_R_use_direct":True,
    "all_signed_exact_source_roots_identical":True,
    "all_R_use_injected_GMP":True,
    "global_NOR46_convolution_function_unchanged":True
}


def preserve(first_path:Path,output:Path):
    original=first_path.read_bytes()
    assert len(original)==RAW_BYTES
    assert sha256(original).hexdigest()==RAW_SHA
    x=json.loads(original)
    assert x["schema"]=="nor48.first_source_locked_signed_program_r_dispatch_cpu.v1"
    assert x["design_issue"]==251 and x["nor47_first_cpu_sha256"]==PRIOR_SHA
    assert x["source_pairs"]==16 and x["new_nor48_source_identities"]==0
    assert x["registered_replicate_blocks"]==[100,104,108,112]
    assert x["repeat_count"]==5 and x["armed_auditors"]==list(ARMS)
    assert x["strict_authorized_program_only_routing"]
    assert x["explicit_reentrant_GMP_convolution_injection"]
    assert x["global_verifier_monkeypatch_used"] is False
    assert x["full_exact_root_included_all_arms"]
    assert x["shared_signatures_session_transport_excluded_all_arms"]
    assert x["gmp_dependency"]["gmpy2_version"]=="2.3.2"
    assert x["original_engineering_gates"]==GATES
    assert x["first_original_engineering_pass"] is False
    rows=x["source_records"]
    assert len(rows)==len({r["source_sha256"] for r in rows})==16
    assert Counter(r["replicate"] for r in rows)=={100:4,104:4,108:4,112:4}
    assert Counter(r["has_R_in_authorized_program"] for r in rows)=={
        True:12,False:4}
    assert all(r["all_four_full_exact_outputs_and_roots_equal"] for r in rows)
    for row in rows:
        timing=row["five_interleaved_arm_ms"]
        assert set(timing)==set(ARMS)
        assert all(len(v)==5 and all(math.isfinite(t) and t>0 for t in v)
                   for v in timing.values())
        assert row["authorized_program_routing"]==(
            "gmp_full_trace" if row["has_R_in_authorized_program"]
            else "nor45_direct")
        med={arm:statistics.median(timing[arm]) for arm in ARMS}
        for arm in ARMS:
            assert math.isclose(med[arm],row["arm_medians_ms"][arm],abs_tol=1e-12)
        assert math.isclose(
            row["dispatcher_over_direct"],
            med["nor48_r_aware_dispatch"]/med["nor45_direct"],rel_tol=1e-12)
    actual=statistics.median(r["dispatcher_over_direct"] for r in rows)
    assert math.isclose(actual,.8182870781348079,abs_tol=1e-12)
    assert math.isclose(
        actual,x["overall"]["median_of_sourcewise_dispatch_over_direct"],abs_tol=1e-12)
    assert actual>=.75
    output.mkdir(parents=True,exist_ok=True)
    rawpath=output/"nor48_first_original_cpu.json"
    assert not rawpath.exists()
    rawpath.write_bytes(original)
    receipt={
        "schema":"nor48.first_original_source_locked_cpu_archive.v1",
        "original_run":FIRST_RUN,"original_artifact_id":FIRST_ARTIFACT_ID,
        "original_raw_bytes":RAW_BYTES,"original_raw_sha256":RAW_SHA,
        "prior_NOR47_original_cpu_sha256":PRIOR_SHA,
        "fixed_historical_source_count":16,
        "new_prospective_source_count":0,
        "five_original_registered_gates":GATES,
        "first_original_engineering_pass":False,
        "original_summary":x["overall"],
        "original_by_R":x["by_R"],
        "original_cpu":x["cpu_model"],
        "original_gmp":x["gmp_dependency"],
        "limitation":"Source-locked historical engineering speed benchmark; signed "
                     "authority is separately checked, signing costs excluded. "
                     "Exact nonsuccinct verification, trusted provider external."
    }
    (output/"nor48_first_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print("NOR48 ORIGINAL BYTES/SHA",RAW_BYTES,RAW_SHA)
    print("NOR48 FIRST ENGINEERING PASS",False)
    print("NOR48 ORIGIN RATIO",actual)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    args=p.parse_args()
    preserve(args.original,args.output_dir)


if __name__=="__main__":
    main()
