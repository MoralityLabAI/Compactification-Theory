"""NOR57 first original SOURCE+JOB-LOG failure auditor, entirely read-only.

The original first CPU process completed 60 printed per-source comparisons
then failed BEFORE writing its first CPU JSON. Preserve that NEGATIVE result
without creating a synthetic first CPU JSON or scoring its five missing gates.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json,re

HERE=Path(__file__).resolve().parent/"results"
ORIGINAL_RUN=44303440474
ORIGINAL_FAILED_CPU_JOB=134799161696
ORIGINAL_SOURCE_ARTIFACT=14425579853
ORIGINAL_FAILED_CPU_ARTIFACT=14426523097
SOURCE_BYTES=1474304
SOURCE_SHA="3eed34d3b80fbae3111d0fcad80df0865df3be78f0077460ac9349647616abe2"
REGISTRATION="f0a6091f918217138d466456c34fcc12ac8f3907"
PATTERN=re.compile(
    r"NOR57 FIRST CPU (\d+) /60 ([a-z0-9-]+) (769|1537|3073) "
    r"trusted route (GMP|NATIVE) native/GMP ([0-9]+\.[0-9]+) "
    r"selector/GMP ([0-9]+\.[0-9]+)"
)

def read_original():
    archive=HERE/"nor57_first_original_failure_hosted.zip"
    receipt=json.loads((HERE/"nor57_first_failure_receipt.json").read_bytes())
    assert receipt["schema"]=="nor57.original_first_failed_before_cpu_json.v1"
    assert receipt["original_run"]==ORIGINAL_RUN
    assert receipt["original_failed_CPU_job"]==ORIGINAL_FAILED_CPU_JOB
    assert receipt["original_source_artifact_id"]==ORIGINAL_SOURCE_ARTIFACT
    assert receipt["original_failed_CPU_artifact_id"]==ORIGINAL_FAILED_CPU_ARTIFACT
    assert receipt["first_registration_sha"]==REGISTRATION
    assert receipt["original_first_source_bytes"]==SOURCE_BYTES
    assert receipt["original_first_source_sha256"]==SOURCE_SHA
    assert receipt["first_cpu_json_was_written"] is False
    assert receipt["first_registered_gate_results"] is None
    assert receipt["first_registered_gate_count"]==5
    assert receipt["original_first_engineering_pass"] is False
    assert receipt["original_status"]=="ENGINEERING_FAIL_REPORTING_EXCEPTION"
    assert archive.stat().st_size==receipt["original_archive_bytes"]
    assert sha256(archive.read_bytes()).hexdigest()==receipt["original_archive_sha256"]
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={
            "nor57_first_original_60_public_only.json",
            "nor57_first_failed_CPU_job_log.txt"
        }
        source=z.read("nor57_first_original_60_public_only.json")
        log=z.read("nor57_first_failed_CPU_job_log.txt")
    assert len(source)==SOURCE_BYTES and sha256(source).hexdigest()==SOURCE_SHA
    assert len(log)==receipt["original_job_log_bytes"]
    assert sha256(log).hexdigest()==receipt["original_job_log_sha256"]
    fixture=json.loads(source)
    assert fixture["schema"]=="nor57.first_public_only_60_authenticated_router_sources.v1"
    assert fixture["first_registration_sha"]==REGISTRATION
    assert fixture["source_domain"]=="NOR57/SOURCE/v1"
    assert fixture["source_count"]==60 and fixture["excluded_original_source_IDs"]==348
    assert fixture["public_only"] and fixture["no_inverse_root_payload_signature_session_or_timing"]
    rows=fixture["cases"]
    assert len(rows)==len({r["source_sha256"] for r in rows})==60
    assert Counter(r["length"] for r in rows)=={769:20,1537:20,3073:20}
    assert Counter(r["stride"] for r in rows)=={1:12,4:36,8:12}
    assert all(set(r)=={
        "case_schema","replicate","length","profile","family","stride",
        "coefficient_magnitude_bits","source_coefficients_decimal","source_sha256"}
        for r in rows)
    text=log.decode("utf-8",errors="strict")
    matches=PATTERN.findall(text)
    assert len(matches)==60
    assert [int(m[0]) for m in matches]==list(range(1,61))
    decisions=Counter(m[3] for m in matches)
    assert decisions=={"GMP":52,"NATIVE":8}
    for source_row,measurement in zip(rows,matches):
        seq,profile,m,route,native_ratio,selector_ratio=measurement
        assert source_row["profile"]==profile and source_row["length"]==int(m)
        f=tuple(int(x) for x in source_row["source_coefficients_decimal"])
        bitwidth=max(abs(x).bit_length() for x in f)
        expected=("NATIVE" if source_row["stride"]==4 and
                  len(f) in (769,1537) and bitwidth==16 else "GMP")
        assert route==expected
        assert float(native_ratio)>0 and float(selector_ratio)>0
    assert "KeyError: 'frozen_selector_decision'" in text
    assert "benchmark_cpu.py" in text
    assert "source record" not in receipt.get("first_registered_gate_results_reason","")
    assert "NOR57 FIRST ORIGINAL CPU RAW BYTES SHA256" not in text or (
        # The workflow echo of the next step may contain it as shell code,
        # but no genuine original CPU SHA print can exist after the crash.
        "NOR57 ORIGINAL FIRST ENGINEERING PASS" not in text)
    assert receipt["printed_per_source_comparisons"]==60
    assert receipt["printed_registered_selector_native_count"]==8
    assert receipt["printed_registered_selector_GMP_count"]==52
    return {
        "first_run":ORIGINAL_RUN,
        "source_sha256":SOURCE_SHA,
        "original_outcome":"ENGINEERING_FAIL_REPORTING_EXCEPTION",
        "first_CPU_json_written":False,
        "registered_five_gates_scored":False,
        "first_executed_source_log_count":len(matches),
        "log_sha256":sha256(log).hexdigest()
    }

if __name__=="__main__":
    print("NOR57 immutable original first reporting failure audit",read_original())
