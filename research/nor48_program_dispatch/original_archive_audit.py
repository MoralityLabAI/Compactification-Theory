"""NOR-48 first original historical CPU failure SHA and statistic audit.

Read-only over original first raw evidence, no timing reruns and no edits to
the registered <0.75 target after observing the first 0.8183 ratio.
"""
from __future__ import annotations
import json,math,statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path

HERE=Path(__file__).resolve().parent
RAW=HERE/"results/nor48_first_original_cpu.json"
RECEIPT=HERE/"results/nor48_first_receipt.json"
FIRST_BYTES=27609
FIRST_SHA="d5cf4b0e3860907eb58c2a37b84ff631f023caf2b69e6e0d0e5a322e2f0b2b99"
PRIOR_SHA="809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530"
FIRST_RUN=44253553267
FIRST_ARTIFACT=14404902340
REPS=(100,104,108,112)
ARMS=("nor45_direct","nor46_python_full_trace","nor47_gmp_full_trace",
      "nor48_r_aware_dispatch")
GATES={
    "dispatcher_median_ratio_under_0p75":False,
    "all_four_no_R_use_direct":True,
    "all_signed_exact_source_roots_identical":True,
    "all_R_use_injected_GMP":True,
    "global_NOR46_convolution_function_unchanged":True
}

def audit():
    buf=RAW.read_bytes()
    assert len(buf)==FIRST_BYTES and sha256(buf).hexdigest()==FIRST_SHA
    g=json.loads(buf)
    receipt=json.loads(RECEIPT.read_bytes())
    assert g["schema"]=="nor48.first_source_locked_signed_program_r_dispatch_cpu.v1"
    assert receipt["schema"]=="nor48.first_original_source_locked_cpu_archive.v1"
    assert receipt["original_run"]==FIRST_RUN
    assert receipt["original_artifact_id"]==FIRST_ARTIFACT
    assert receipt["original_raw_bytes"]==FIRST_BYTES
    assert receipt["original_raw_sha256"]==FIRST_SHA
    assert receipt["prior_NOR47_original_cpu_sha256"]==PRIOR_SHA
    assert g["nor47_first_cpu_sha256"]==PRIOR_SHA
    assert g["source_pairs"]==receipt["fixed_historical_source_count"]==16
    assert g["new_nor48_source_identities"]==receipt["new_prospective_source_count"]==0
    assert g["registered_replicate_blocks"]==list(REPS)
    assert g["repeat_count"]==receipt["five_repeats_per_verifier_arm"]==5 if "five_repeats_per_verifier_arm" in receipt else g["repeat_count"]==5
    assert g["armed_auditors"]==list(ARMS)
    assert g["gmp_dependency"]["gmpy2_version"]=="2.3.2"
    assert g["global_verifier_monkeypatch_used"] is False
    assert g["explicit_reentrant_GMP_convolution_injection"]
    assert g["strict_authorized_program_only_routing"]
    assert g["full_exact_root_included_all_arms"]
    assert g["shared_signatures_session_transport_excluded_all_arms"]
    assert g["original_engineering_gates"]==receipt["five_original_registered_gates"]==GATES
    assert g["first_original_engineering_pass"] is receipt["first_original_engineering_pass"] is False
    rows=g["source_records"]
    assert len(rows)==len({x["source_sha256"] for x in rows})==16
    assert Counter(x["replicate"] for x in rows)=={r:4 for r in REPS}
    assert Counter(x["has_R_in_authorized_program"] for x in rows)=={True:12,False:4}
    for x in rows:
        assert x["all_four_full_exact_outputs_and_roots_equal"]
        times=x["five_interleaved_arm_ms"]
        assert set(times)==set(ARMS)
        for arm in ARMS:
            assert len(times[arm])==5
            assert all(math.isfinite(v) and v>0 for v in times[arm])
            assert math.isclose(statistics.median(times[arm]),
                                x["arm_medians_ms"][arm],rel_tol=1e-13)
        assert math.isclose(x["dispatcher_over_direct"],x["arm_medians_ms"][
            "nor48_r_aware_dispatch"]/x["arm_medians_ms"]["nor45_direct"],
            rel_tol=1e-13)
        assert x["authorized_program_routing"]==(
            "gmp_full_trace" if x["has_R_in_authorized_program"]
            else "nor45_direct")
    primary=statistics.median(x["dispatcher_over_direct"] for x in rows)
    assert math.isclose(primary,.8182870781348079,abs_tol=1e-14)
    assert math.isclose(primary,g["overall"][
        "median_of_sourcewise_dispatch_over_direct"],abs_tol=1e-14)
    assert receipt["original_summary"]==g["overall"]
    assert receipt["original_by_R"]==g["by_R"]
    assert primary>=.75
    return {
        "schema":"nor48.original_first_immutable_CPU_audit.v1",
        "first_run":FIRST_RUN,
        "original_first_sha256":FIRST_SHA,
        "historical_source_count":16,
        "gmp_R_sources":12,
        "original_median_dispatch_over_direct":primary,
        "registered_threshold":.75,
        "engineering_pass":False,
        "original_gates":GATES,
        "no_retrospective_rebenchmark":True,
    }

if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
