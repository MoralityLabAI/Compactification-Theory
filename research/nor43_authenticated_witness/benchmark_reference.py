"""NOR-43 exploratory CPU verification timing on SHA-archived NOR42 pairs.

Not a registered science endpoint and not part of the first original smoke.
Measures verified reference manifest signature and EXACT challenged-index k16
Merkle opening checks on 16 already completed NOR42 source identities.
Tree construction, proposal generation, source reconstruction and JSON
transport are excluded from the latency measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from research.nor43_authenticated_witness import historical_smoke as source
from research.nor43_authenticated_witness import signed_witness as producer
from research.nor43_authenticated_witness import verifier_reference as verifier

SELECT=(0,1,2,3,8,9,10,11,20,21,22,23,32,33,34,35)
REPEATS=24
WARMUPS=3


def percentile(values,p):
    vals=sorted(values)
    return vals[min(len(vals)-1,max(0,math.ceil(p*len(vals))-1))]


def bench():
    fixture,graded=source.original_first_archive()
    frozen=json.loads((source.ROOT/"research/nor43_authenticated_witness/results/"
                       "nor43_archived_64_merkle.json").read_bytes())
    assert len(frozen["source_records"])==64
    report=[]
    pinned=producer.test_reference_public_key()
    for j in SELECT:
        row=fixture["cases"][j]
        original=graded["source_records"][j]
        frozen_row=frozen["source_records"][j]
        src=row["full_public_input_sha256"]
        assert src==original["public_sha256"]==frozen_row["original_nor42_public_sha256"]
        pred=source.nor42.nor37.public_proposal(row)["pred"]
        _,hidden,_,_=source.nor42.restore_worlds(row,pred)
        assert source.nor42.nor24.sha_target(hidden)==original[
            "private_hidden_truth_sha256_after_freeze"]
        tree=producer.OutputTree(hidden)
        assert tree.root.hex()==frozen_row["true_hidden_root_sha256"]
        signed=producer.signed_reference_manifest(tree,src,row["degree"],signing_key=producer.test_private_key())
        checked=tuple(frozen_row["historical_k16_checked_indices"])
        opening=[tree.opening(i) for i in checked]
        verified=verifier.verify_signed_manifest(
            signed,pinned,src,row["degree"],len(hidden),producer.PROVIDER_ID)
        should_detect=bool(set(checked)&{
            i for i,(a,b) in enumerate(zip(pred,hidden)) if a!=b})
        assert verifier.verified_sample_rejection(
            pred,verified,opening,challenged_indices=checked)==should_detect
        signatures=[];proofs=[]
        for rep in range(REPEATS+WARMUPS):
            t0=time.perf_counter_ns()
            auth=verifier.verify_signed_manifest(
                signed,pinned,src,row["degree"],len(hidden),producer.PROVIDER_ID)
            t1=time.perf_counter_ns()
            result=verifier.verified_sample_rejection(
                pred,auth,opening,challenged_indices=checked)
            t2=time.perf_counter_ns()
            assert result==should_detect
            if rep>=WARMUPS:
                signatures.append((t1-t0)/1000)
                proofs.append((t2-t1)/1000)
        report.append({
            "historical_source_sha256":src,
            "observation_degree":row["degree"],
            "input_family":row["family"],
            "coefficient_count":len(hidden),
            "merkle_depth":tree.depth,
            "repeat_count":REPEATS,
            "median_signature_and_manifest_verify_microseconds":statistics.median(signatures),
            "p95_signature_and_manifest_verify_microseconds":percentile(signatures,.95),
            "median_16_openings_verify_microseconds":statistics.median(proofs),
            "p95_16_openings_verify_microseconds":percentile(proofs,.95),
            "median_total_verify_microseconds":statistics.median([
                a+b for a,b in zip(signatures,proofs)]),
        })
        print(f"NOR43 benchmark source {len(report)}/{len(SELECT)}",flush=True)
    assert len(report)==len(SELECT)==16
    stats={
        "median_signature_verification_microseconds":statistics.median(
            r["median_signature_and_manifest_verify_microseconds"] for r in report),
        "median_16_openings_verification_microseconds":statistics.median(
            r["median_16_openings_verify_microseconds"] for r in report),
        "median_total_verification_microseconds":statistics.median(
            r["median_total_verify_microseconds"] for r in report),
    }
    return {
        "schema":"nor43.historical_reference_verify_timing_exploratory.v1",
        "original_first_smoke_sha256":source.sha256(
            (source.ROOT/"research/nor43_authenticated_witness/results/"
             "nor43_archived_64_merkle.json").read_bytes()).hexdigest(),
        "historical_source_pairs":16,
        "new_nor43_source_identifiers":0,
        "selection":"fixed zero-based indices "+",".join(map(str,SELECT)),
        "repetitions_per_pair":REPEATS,
        "python_version":platform.python_version(),
        "cryptography_version":version("cryptography"),
        "operating_system":platform.platform(),
        "cpu_count_reported":os.cpu_count(),
        "cpu_model":_cpu_model(),
        "summary":stats,
        "per_source":report,
        "limitations":"Exploratory wall-clock CPU performance on hosted CI runner. "
                      "Excludes output calculation, Merkle tree construction, "
                      "challenge entropy generation and network serialization. "
                      "Public test keys and historical source data only."
    }


def _cpu_model():
    try:
        with open("/proc/cpuinfo",encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":",1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unreported"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=bench()
    raw=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(raw,encoding="utf-8")
    print("NOR43 HOSTED DESCRIPTIVE VERIFY TIMES",result["summary"])
    print("NOR43 BENCHMARK PYTHON CRYPTO CPU",
          result["python_version"],result["cryptography_version"],
          result["cpu_model"])


if __name__=="__main__":
    main()
