"""NOR57 one-shot first original reporting FAIL source/log evidence publisher.

The originally registered CPU process computed and printed 60 source-level
signed comparisons but crashed in summarize BEFORE writing any CPU JSON.
Store original public bytes and exact GitHub job log; DO NOT manufacture a
CPU JSON, score uncomputed gates, or call an altered runner as original.
"""
from __future__ import annotations
import json,re
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

from research.nor57_authenticated_router.original_failure_audit import (
    SOURCE_BYTES,SOURCE_SHA,REGISTRATION,PATTERN
)
ROOT=Path(__file__).resolve().parent
SOURCE=Path("/tmp/nor57_original_source/nor57_first_original_60_public_only.json")
LOG=Path("/tmp/nor57_first_failed_CPU_job_log.txt")

def main():
    dest=ROOT/"results"
    archive=dest/"nor57_first_original_failure_hosted.zip"
    if archive.exists():
        raise RuntimeError("original first failure ZIP already exists, cannot replace")
    public=SOURCE.read_bytes()
    logs=LOG.read_bytes()
    assert len(public)==SOURCE_BYTES and sha256(public).hexdigest()==SOURCE_SHA
    x=json.loads(public)
    assert x["first_registration_sha"]==REGISTRATION and x["source_count"]==60
    assert x["no_inverse_root_payload_signature_session_or_timing"]
    text=logs.decode("utf-8",errors="strict")
    hits=PATTERN.findall(text)
    assert len(hits)==60 and [int(h[0]) for h in hits]==list(range(1,61))
    assert sum(h[3]=="NATIVE" for h in hits)==8
    assert sum(h[3]=="GMP" for h in hits)==52
    assert "KeyError: 'frozen_selector_decision'" in text
    assert "benchmark_cpu.py" in text and "summarize" in text
    dest.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in (
            ("nor57_first_original_60_public_only.json",public),
            ("nor57_first_failed_CPU_job_log.txt",logs)
        ):
            info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    receipt={
        "schema":"nor57.original_first_failed_before_cpu_json.v1",
        "original_run":44303440474,
        "original_failed_CPU_job":134799161696,
        "original_source_artifact_id":14425579853,
        "original_failed_CPU_artifact_id":14426523097,
        "first_registration_sha":REGISTRATION,
        "original_first_source_bytes":SOURCE_BYTES,
        "original_first_source_sha256":SOURCE_SHA,
        "original_job_log_bytes":len(logs),
        "original_job_log_sha256":sha256(logs).hexdigest(),
        "original_archive_bytes":archive.stat().st_size,
        "original_archive_sha256":sha256(archive.read_bytes()).hexdigest(),
        "first_cpu_json_was_written":False,
        "first_registered_gate_count":5,
        "first_registered_gate_results":None,
        "first_registered_gate_results_reason":"Result aggregation KeyError before five-gate scoring.",
        "original_first_engineering_pass":False,
        "original_status":"ENGINEERING_FAIL_REPORTING_EXCEPTION",
        "printed_per_source_comparisons":60,
        "printed_registered_selector_native_count":8,
        "printed_registered_selector_GMP_count":52,
        "first_reproduction_is_not_a_valid_original_result":True,
        "limitation":"Original source-level comparison logs were rounded to four decimals; full five-repeat JSON never existed. Any later replay is a separately labeled follow-up."
    }
    (dest/"nor57_first_failure_receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    report=f"""# NOR-57 original first source-locked CPU run: engineering FAIL (result writer exception)

**Original first outcome: ENGINEERING FAIL, with five registered performance/correctness gates unscored.** [Original GitHub Actions #44303440474](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44303440474) froze the 60 new source-only identities upstream of the first CPU process. The first CPU job **#134799161696** executed and printed all 60 signed witness/source comparisons using the registered fixed 8 native/52 GMP selector. Its reporter then raised `KeyError: 'frozen_selector_decision'` because first source records used `frozen_public_selector_decision`. The exception occurred at `benchmark_cpu.py` in `summarize`, **before the original first raw CPU JSON and five Boolean gate verdicts could be written**. No complete first CPU JSON exists. The five gates cannot be scored from the rounded stdout lines.

## Original immutable first evidence

- Registration [`f0a6091f`](https://github.com/MoralityLabAI/Compactification-Theory/commit/f0a6091f918217138d466456c34fcc12ac8f3907), before fresh source generation or timing.
- Upstream public-only source roster: **{SOURCE_BYTES:,} exact bytes**, SHA256 `{SOURCE_SHA}`, artifact **#14425579853**. All 60 source SHA IDs are disjoint from 348 old first original NOR51–56 IDs.
- Original first CPU job log: **{len(logs):,} exact bytes**, SHA256 `{receipt['original_job_log_sha256']}`, job #134799161696. All 60 printed source rows and exact original reporting failure are preserved.
- The failed CPU Actions artifact **#14426523097** contains the source JSON alone, because **no original CPU JSON was written**.
- Both source and first job log are stored byte-identically in [nor57_first_original_failure_hosted.zip](results/nor57_first_original_failure_hosted.zip), with archive SHA256 `{receipt['original_archive_sha256']}` pinned by [first failure receipt](results/nor57_first_failure_receipt.json) and permanently checked by [read-only first original failure audit](original_failure_audit.py).

## Original registered five gates

| Original preregistered gate | First score |
| --- | --- |
| Ratio of sums of all 60 selector/GMP source-median full signed times <0.95× | **UNSCORED** |
| Exactly 8 native-selected sourcewise median native/GMP <0.85× | **UNSCORED** |
| Exactly 52 GMP-selected sourcewise median selector/GMP <1.10× | **UNSCORED** |
| All 60 full signed exact outputs, roots and hostile witness controls | **UNSCORED from first final evidence** |
| First source SHA, 348 exclusions, original caps/repeats and immutable result | **FAIL: first full CPU JSON missing** |

**Engineering verdict: FAIL (unscored first run).** Predata toy/security controls passed, and the 60 source-level log prints followed the exact selector path; those observations provide useful debugging evidence. Rounded log ratios are insufficient to reconstruct the full five original gate values, five original timing samples and exact reference Merkle roots. Preserving original scientific integrity requires retaining this first negative result as is.

## Post-error comparison and scope

The reporter's source record key mismatch can be fixed without changing the underlying original arithmetic/routing rule. Any rerun is explicitly a **new follow-up run**, preferably on fresh SHA-disjoint source identities preregistered before exposure. It cannot be relabeled the first NOR57 run, and its results cannot retroactively score this first source. The signed native/GMP arithmetic remains a synthetic, nonsuccinct exact inverse witness verifier using an intentionally public Ed25519 research signing fixture. m3073 remains outside the NOR45 signed-program authority grammar; genuine remote provider honesty and transport economics are external.
"""
    (ROOT/"RESULTS.md").write_text(report)
    print("NOR57 immutable first original public source SHA",SOURCE_SHA)
    print("NOR57 immutable first original CPU job log SHA",receipt["original_job_log_sha256"])
    print("NOR57 original first engineering FAIL, five original gates UNSCORED")
    print("NOR57 original first failure ZIP SHA",receipt["original_archive_sha256"])

if __name__=="__main__":
    main()
