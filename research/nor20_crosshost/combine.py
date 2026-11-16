"""NOR-20 independent public fixture and paired Linux/Windows receipt audit.

Reads the identical source fixture and two host-specific raw timing files.
Reconstructs all original native exact masks and candidate query-output
unanimity predicates using RRC25. Independently verifies time sample
medians, reported paired speed/memory ratios, and width-stratum counts.

Host-specific timing and Python allocations remain physical observations.
This code never creates any withheld query output or loads a private
generating program as a model policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.rrc_compactification import compound25 as corpus
from research.nor17_prefix_certificates import experiment as nor17

WIDTHS=(4,8,12,16)
FAMILIES=("wide_dense","wide_sparse","wide_late","wide_rational")
MODES=("independent16","cached16","stream16_queryfirst",
       "stream16_verifyfirst","chunk4_verifyfirst")
PRIMARY="stream16_verifyfirst"
OLD_SHA="9905aa23466b98ba2068e6daf9206d35c453b8b3b75bd27c3ab7ff8f0db4041b"


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-8)


def public_case(entry):
    h=entry["public_header"]
    hdr=corpus.Header(h["base"],h["delay"],h["spacing"],h["amplitude"])
    demos=tuple((tuple(inp),tuple(out)) for inp,out in entry["demos"])
    assert len(demos)==2 and all(len(inp)==len(out)==193
                                 for inp,out in demos)
    query=tuple(entry["public_query_input"])
    assert len(query)==257 and query[0]==1
    c=corpus.Case("nor20-independent",entry["family"],demos,
                  query,(),None)
    assert c.future==() and c.program is None
    source_sha=sha256(json.dumps([c.visible,c.query],
                          separators=(",",":")).encode()).hexdigest()
    assert source_sha==entry["full_public_sha256"]
    return c,hdr


def native_masks(c,h):
    bits=0
    for j in range(16):
        if corpus.verify_template(c,h,corpus.OPS[j//4],corpus.OPS[j%4]):
            bits|=1<<j
    assert bits
    hypothetical=nor17.candidate_outputs(c.query,h,corpus.FUTURE)
    unanimous=len({hypothetical[j] for j in range(16) if bits&(1<<j)})==1
    return bits,unanimous


def evaluate_one_host(fixture,result,expected_host,fixture_sha):
    assert result["schema"]=="nor20.wide_coefficients_cross_host_cpu_tracemalloc.v1"
    assert result["fixture_schema"]==fixture["schema"]
    assert result["fixture_sha256"]==fixture_sha
    assert result["source_seed_offset"]==94120000
    assert result["nominal_input_widths"]==list(WIDTHS)
    assert result["public_source_families"]==list(FAMILIES)
    assert result["cases"]==len(result["records"])==64
    assert result["cohorts"]==4 and result["trials"]==7
    assert result["warmups"]==2
    assert result["modes"]==list(MODES)
    assert result["primary_mode"]==PRIMARY
    assert result["host"]["host_label"]==expected_host
    assert result["host"]["cpu_count"]>=1
    if expected_host=="windows_2025":
        assert "Windows" in result["host"]["os"]
    else:
        assert "Linux" in result["host"]["os"]
    for source,rec in zip(fixture["cases"],result["records"],strict=True):
        c,h=public_case(source)
        truth=native_masks(c,h)
        assert truth==(source["original_exact_accept_mask"],
                      source["original_query_output_unanimity"])
        assert rec["public_source_sha256"]==source["full_public_sha256"]
        assert (rec["cohort"],rec["family"],rec["nominal_input_width"])==(
            source["cohort"],source["family"],source["width_bits_seed"])
        assert rec["mask_and_certificate_parity"]
        assert (rec["reference_mask"],rec["reference_unanimous"])==truth
        assert rec["actual_public_source_bits"]==source[
            "actual_max_public_source_bits"]
        assert rec["actual_public_output_bits"]==source[
            "actual_max_public_output_bits"]
        assert rec["max_public_reciprocal_bits"]>=1
        assert set(rec["samples_ns"])==set(MODES)
        assert set(rec["median_ns"])==set(MODES)
        assert set(rec["tracemalloc_complete"])==set(MODES)
        assert set(rec["tracemalloc_verifier_only"])==set(MODES)
        assert set(rec["inverse_ledger"])==set(MODES)
        for mode in MODES:
            samples=rec["samples_ns"][mode]
            assert len(samples)==7
            for sample in samples:
                assert set(sample)=={"query_ns","verification_ns","certificate_ns","total_ns"}
                assert sample["total_ns"]==(
                    sample["query_ns"]+sample["verification_ns"]+
                    sample["certificate_ns"])
                assert sample["total_ns"]>0
            for key in ("query_ns","verification_ns","certificate_ns","total_ns"):
                assert close(rec["median_ns"][mode][key],
                             statistics.median(z[key] for z in samples))
            mem=rec["tracemalloc_complete"][mode]
            assert mem["mask"]==truth[0]
            assert mem["python_peak_allocation_bytes"]>=mem[
                "python_current_allocation_bytes"]>=0
            ver=rec["tracemalloc_verifier_only"][mode]
            assert ver["mask"]==truth[0]
            assert ver["python_verifier_only_peak_bytes"]>=0
            assert rec["inverse_ledger"][mode]["R_integer_products"]>0
            assert rec["inverse_ledger"][mode]["R_operand_bitwidth_product_proxy"]>0
        a=rec["tracemalloc_complete"][PRIMARY]["python_peak_allocation_bytes"]
        b=rec["tracemalloc_complete"]["cached16"]["python_peak_allocation_bytes"]
        speed=(rec["median_ns"]["independent16"]["total_ns"]/
               rec["median_ns"][PRIMARY]["total_ns"])
        assert close(rec["primary_stream_to_cached_peak_ratio"],a/b)
        assert close(rec["primary_native_to_stream_cpu_ratio"],speed)
        assert rec["primary_memory_win"]==(a<b)
    rows=result["records"]
    assert len({r["public_source_sha256"] for r in rows})==64
    assert Counter(r["nominal_input_width"] for r in rows)=={
        b:16 for b in WIDTHS}
    assert Counter(r["family"] for r in rows)=={
        family:16 for family in FAMILIES}
    assert Counter(r["cohort"] for r in rows)=={
        cohort:16 for cohort in range(4)}
    med_mem=statistics.median(z["primary_stream_to_cached_peak_ratio"] for z in rows)
    med_speed=statistics.median(z["primary_native_to_stream_cpu_ratio"] for z in rows)
    wins=sum(z["primary_memory_win"] for z in rows)
    expected_pass=(med_mem<=.8 and med_speed>=2 and wins>=48)
    assert close(med_mem,result["overall"]["median_peak_ratio_stream_over_cache"])
    assert close(med_speed,result["overall"]["median_native_to_stream_cpu_ratio"])
    assert wins==result["overall"]["stream_lower_python_peak_count"]
    assert result["primary_pass"]==expected_pass
    for width in WIDTHS:
        group=[r for r in rows if r["nominal_input_width"]==width]
        assert len(group)==result["by_width"][str(width)]["cases"]==16
    for family in FAMILIES:
        group=[r for r in rows if r["family"]==family]
        assert len(group)==result["by_family"][family]["cases"]==16
    for mode,mem in result["rss_separate_process"].items():
        assert mode in MODES and mem["cases"]==64 and mem["mode"]==mode
        if expected_host=="windows_2025":
            assert mem["delta_peak_KiB"] is None
    assert result["primary_memory_block_95"]["draws"]==4096
    assert result["primary_cpu_block_95"]["draws"]==4096
    return {"label":expected_host,
            "median_memory_ratio":med_mem,
            "lower_memory_cases":wins,
            "median_native_cpu_speedup":med_speed,
            "passed":expected_pass,
            "median_max_reciprocal_bits_by_nominal_width":{
                str(width):statistics.median(
                    r["max_public_reciprocal_bits"] for r in rows
                    if r["nominal_input_width"]==width)
                for width in WIDTHS}}


def compare(fixture_path,linux_path,windows_path):
    raw=Path(fixture_path).read_bytes()
    fixture=json.loads(raw)
    assert fixture["schema"]=="nor20.public_only_wide_coefficients_source_v1"
    assert fixture["original_NOR18_SHA256_excluded"]==OLD_SHA
    assert fixture["private_program_identities_and_future_query_targets_absent"]
    assert fixture["public_fixture_only"]
    assert fixture["nominal_widths"]==list(WIDTHS)
    assert fixture["public_families"]==list(FAMILIES)
    assert len(fixture["cases"])==64
    fixture_sha=sha256(raw).hexdigest()
    linux=evaluate_one_host(fixture,json.loads(Path(linux_path).read_bytes()),
                            "linux_ubuntu_2404",fixture_sha)
    windows=evaluate_one_host(fixture,json.loads(Path(windows_path).read_bytes()),
                              "windows_2025",fixture_sha)
    return {
        "schema":"nor20.two_host_byte_identical_wide_source_comparison.v1",
        "public_fixture_sha256":fixture_sha,
        "two_host_source_case_count":64,
        "source_cohorts":4,
        "native_reference_masks_recomputed":64*2,
        "all_cases_per_nominal_width":16,
        "linux":linux,"windows":windows,
        "both_hosts_preregistered_passed":linux["passed"] and windows["passed"],
        "scope":{
            "same_byte_identical_fixture_both_hosts":True,
            "two_operating_systems_not_guaranteed_two_distinct_CPU_microarchitectures":True,
            "future_query_labels_and_planted_program_absent":True,
            "cohort_bootstrap_conditional_on_64_generated_sources":True,
        },
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",required=True,type=Path)
    parser.add_argument("--linux",required=True,type=Path)
    parser.add_argument("--windows",required=True,type=Path)
    parser.add_argument("--out",type=Path)
    p=parser.parse_args()
    result=compare(p.fixture,p.linux,p.windows)
    encoded=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
    if p.out:
        p.out.parent.mkdir(parents=True,exist_ok=True)
        p.out.write_bytes(encoded)
    print("NOR20 CROSS HOST SOURCE AND TIMER AUDIT",{
        "fixture":result["public_fixture_sha256"],
        "cases":result["two_host_source_case_count"],
        "linux":result["linux"],
        "windows":result["windows"],
        "both_hosts_success":result["both_hosts_preregistered_passed"],
    })


if __name__=="__main__":
    main()
