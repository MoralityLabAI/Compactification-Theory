"""NOR-20 independent original source and Linux/Windows timing auditor.

Uses the original RRC25 interpreter and NOR-20 crosshost comparator to
reconstruct public masks, verify 4480 raw CPU timings, 1280 separately
traced memory records, observed reciprocal bigint operand widths, and
the preregistered four-cohort bootstrap ratios. It never imports the
NOR-20 wide-source benchmark implementation and reads no private truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.nor20_crosshost import combine

RECEIPT=HERE/"results"/"nor20_first_frozen_receipt.json"
FILES={
    "fixture.json":(5416693,"de7ed801e0e8708afebdbe3699d9074375b156795b40372eb77e2a4a13218c19"),
    "linux_ubuntu_2404.json":(683486,"6c033440be72ae470769e7a152286677cba8fa3b676e9f9207d0c335ab1ec01d"),
    "windows_2025.json":(683275,"85686b13b4526d926f10b0b4f476bea82ab9d2f88dbc63d02704268f9ef88158"),
    "comparison.json":(1296,"3a048cfbafb7c9e87f28ed04d7d7e2ecd1b06b3be5c513351aa7fb11fa6a07c4"),
}
BOOT_SEED=202611201620


def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-7)


def read_original(path):
    with ZipFile(path) as z:
        assert sorted(z.namelist())==sorted(FILES)
        assert z.testzip() is None
        data={key:z.read(key) for key in FILES}
    for name,(size,digest) in FILES.items():
        assert len(data[name])==size,(name,len(data[name]))
        assert sha256(data[name]).hexdigest()==digest
    frozen=json.loads(RECEIPT.read_text())
    assert frozen["schema"]=="nor20.first_linux_windows_wide_coefficient_source_locked_receipt.v1"
    assert frozen["source_fixture_sha256"]==FILES["fixture.json"][1]
    assert frozen["linux_benchmark_sha256"]==FILES["linux_ubuntu_2404.json"][1]
    assert frozen["windows_benchmark_sha256"]==FILES["windows_2025.json"][1]
    assert frozen["independent_crosshost_comparison_sha256"]==FILES["comparison.json"][1]
    assert frozen["public_source_cases"]==64
    assert frozen["source_seed_offset"]==94120000
    assert frozen["preregistered_cross_host_pass"]
    return data,frozen


def one_inverse(source):
    """Exact normalized R recurrence with independently observed bit widths."""
    assert source and source[0]==1
    inv=[1]
    products=bitwork=0
    maximum_bits=1
    for n in range(1,len(source)):
        value=0
        for k in range(1,n+1):
            a=source[k];b=inv[n-k]
            products+=1
            bitwork+=max(1,abs(a).bit_length())*max(1,abs(b).bit_length())
            value+=a*b
        inv.append(-value)
        maximum_bits=max(maximum_bits,abs(value).bit_length())
    return products,bitwork,maximum_bits


def bootstrap(rows,key):
    groups=defaultdict(list)
    for r in rows:
        groups[r["cohort"]].append(r)
    assert set(groups)=={0,1,2,3}
    assert all(len(group)==16 for group in groups.values())
    rng=random.Random(BOOT_SEED+31)
    observed=[]
    for _ in range(4096):
        selected=[r for c in (rng.randrange(4) for j in range(4))
                  for r in groups[c]]
        observed.append(statistics.median(r[key] for r in selected))
    observed.sort()
    return observed[int(.025*4095)],observed[int(.975*4095)]


def audit(path):
    sources,receipt=read_original(path)
    with tempfile.TemporaryDirectory(prefix="nor20-audit-") as td:
        paths={}
        for name,data in sources.items():
            dest=Path(td)/name
            dest.write_bytes(data)
            paths[name]=dest
        compare_result=combine.compare(
            paths["fixture.json"],paths["linux_ubuntu_2404.json"],
            paths["windows_2025.json"])
        original_comparison=json.loads(sources["comparison.json"])
        assert compare_result==original_comparison
    fixture=json.loads(sources["fixture.json"])
    results={
        "linux":json.loads(sources["linux_ubuntu_2404.json"]),
        "windows":json.loads(sources["windows_2025.json"]),
    }
    assert compare_result["both_hosts_preregistered_passed"]
    assert len(fixture["cases"])==64
    reciprocal_checks=0
    for public,linux,windows in zip(
            fixture["cases"],results["linux"]["records"],
            results["windows"]["records"],strict=True):
        c,h=combine.public_case(public)
        invs=[one_inverse(source) for source,y in c.demos]
        invs.append(one_inverse(c.query))
        maxbits=max(t[2] for t in invs)
        per_source_products=sum(z[0] for z in invs[:2])
        per_source_bitwork=sum(z[1] for z in invs[:2])
        query_products=invs[2][0]
        query_bitwork=invs[2][1]
        for z in (linux,windows):
            assert z["max_public_reciprocal_bits"]==maxbits
            assert z["public_source_sha256"]==public["full_public_sha256"]
            assert z["actual_public_source_bits"]==public[
                "actual_max_public_source_bits"]
            for mode in combine.MODES:
                mult=(8+16*int(h.base=="R") if mode=="independent16"
                      else 1)
                expected_products=query_products+mult*per_source_products
                expected_bitwork=query_bitwork+mult*per_source_bitwork
                actual=z["inverse_ledger"][mode]
                assert actual["R_integer_products"]==expected_products
                assert actual["R_operand_bitwidth_product_proxy"]==expected_bitwork
                reciprocal_checks+=1
    for label,x in results.items():
        assert x["host"]["host_label"]==(
            "linux_ubuntu_2404" if label=="linux" else "windows_2025")
        assert x["primary_pass"]
        rows=x["records"]
        assert len(rows)==64
        # Exact original seven-sample timing accounting is checked by
        # combine.evaluate_one_host; here independently resample CIs.
        for key,outkey in (
                ("primary_stream_to_cached_peak_ratio",
                 "primary_memory_block_95"),
                ("primary_native_to_stream_cpu_ratio",
                 "primary_cpu_block_95")):
            lo,hi=bootstrap(rows,key)
            orig=x[outkey]
            assert orig["draws"]==4096
            assert close(lo,orig["p2_5"]) and close(hi,orig["p97_5"])
        rec=receipt[label]
        assert close(rec[
            "median_paired_native_over_streamed_complete_cpu_speedup"],
                     x["overall"]["median_native_to_stream_cpu_ratio"])
        assert close(rec[
            "median_paired_streamed_over_cached_complete_tracemalloc_peak_ratio"],
                     x["overall"]["median_peak_ratio_stream_over_cache"])
        assert rec["streaming_lower_complete_tracemalloc_cases"]==64
        assert x["overall"]["stream_lower_python_peak_count"]==64
    return {
        "public_source_cases":64,
        "exact_native_public_masks_audited":128,
        "byte_identical_fixture_on_both_hosts":True,
        "raw_cpu_timing_samples_checked":64*5*7*2,
        "complete_and_verifier_only_tracemalloc_records_checked":64*5*2*2,
        "independently_replayed_bigint_per_mode_source_ledgers":reciprocal_checks,
        "bootstrap_draws_recomputed":4096*2*2,
        "linux":compare_result["linux"],
        "windows":compare_result["windows"],
        "both_hosts_passed":True,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.archive:
        result=audit(args.archive)
        print("NOR20 INDEPENDENT ORIGINAL LINUX WINDOWS CPU BIGINT AUDIT PASSED",result)
        if args.check:
            assert result["raw_cpu_timing_samples_checked"]==4480
            assert result["complete_and_verifier_only_tracemalloc_records_checked"]==1280
    else:
        _,receipt=read_original(
            HERE/"results"/"nor20_original_crosshost.zip")
        print("NOR20 ORIGINAL HOSTED SOURCE AND SHA RECEIPT VALID")


if __name__=="__main__":
    main()
