"""Copy the observed first hosted ZIP byte-for-byte and record its provenance.

No source, signature, output or timing is generated here. Expected raw hashes
were observed in first-run logs and independently downloaded artifact bytes.
This script refuses existing archive/receipt destinations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

RUN = 44327994013
REGISTRATION = "363ef19abeba68e2897c593697ad56e2aaf4fefc"
TRIGGER = "a7edfa6850206fc01a9a67650e09ddd31d198766"
ZIP_SHA = "c92e2c4e3bfcfd8e1371d1504c89d6812ac263ef270fc9d62e89d335f23da53c"
MEMBERS = {
    "nor60_first_original_60_public_only.json": (1021038, "b44655796bdc4c9ed47fade2cd9cd563985d0b40ca1177856bb45f36d17185fd"),
    "nor60_first_original_cpu.json": (27720732, "fba38d6346d35b371464c1d811360879918aeccc0a8932538b6f3206cc4bc52d"),
    "nor60_first_original_progress.jsonl": (898921, "48828148a4dbce8d53e86002f5029902f4e80857ec6f53eddd25c09b3fe21dea"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def archive_first(source_zip, directory):
    source_zip, directory = Path(source_zip), Path(directory)
    target = directory / "nor60_first_original_hosted.zip"
    receipt_path = directory / "nor60_first_receipt.json"
    require(not target.exists() and not receipt_path.exists(), "first evidence already exists")
    raw_zip = source_zip.read_bytes()
    require(len(raw_zip) == 12310423 and hashlib.sha256(raw_zip).hexdigest() == ZIP_SHA,
            "downloaded original hosted ZIP differs from observed first artifact")
    raw = {}
    with ZipFile(source_zip) as z:
        require(z.testzip() is None and len(z.namelist()) == 3 and set(z.namelist()) == set(MEMBERS),
                "original archive integrity or inventory changed")
        for name, (size, digest) in MEMBERS.items():
            raw[name] = z.read(name)
            require(len(raw[name]) == size and hashlib.sha256(raw[name]).hexdigest() == digest,
                    "original member differs: " + name)
    cpu = json.loads(raw["nor60_first_original_cpu.json"])
    source = json.loads(raw["nor60_first_original_60_public_only.json"])
    require(cpu["first_registration_sha"] == source["first_registration_sha"] == REGISTRATION,
            "first registration changed")
    require(cpu["github_run_id"] == str(RUN) and cpu["github_run_attempt"] == "1" and
            cpu["github_sha"] == TRIGGER, "first-run provenance changed")
    gates = cpu["first_original_five_registered_gates"]
    primary = {
        "sparse40_interpreter_median": cpu["primary_forty_sparse"]["median_sourcewise_ratio_to_original"]["optimized_interpreter_full_root"],
        "all60_combined_ratio_of_sums": cpu["overall"]["source_median_ratio_of_sums_to_original"]["optimized_interpreter_warm_root"],
        "profile4_12_interpreter_median": cpu["primary_twelve_profile4"]["median_sourcewise_ratio_to_original"]["optimized_interpreter_full_root"],
    }
    receipt = {
        "schema": "nor60.first_original_60_signed_NOR45_CPU_archive.v1",
        "original_first_registration_commit": REGISTRATION,
        "original_trigger_commit": TRIGGER, "original_first_run": RUN,
        "original_source_artifact_id": 14438371253, "original_CPU_artifact_id": 14436529899,
        "original_source_job_id": 134884681802, "original_CPU_job_id": 134884822335,
        "upstream_source_uploaded_at": "2026-11-18T14:47:17Z",
        "source_job_completed_at": "2026-11-18T14:47:19Z",
        "cpu_job_started_at": "2026-11-18T14:47:21Z",
        "cpu_job_completed_at": "2026-11-18T14:50:39Z",
        "CPU_artifact_uploaded_at": "2026-11-18T14:50:36Z",
        "original_source_artifact_zip_sha256": "bea12212a0696ab1b5ad85c3e901f3ed17663e3c9627fd615cbb6df44dd25e03",
        "original_source_artifact_zip_bytes": 73788,
        "archive_zip_sha256": ZIP_SHA, "archive_zip_bytes": len(raw_zip),
        "archive_is_byte_identical_original_hosted_artifact": True,
        "raw_public_source_sha256": MEMBERS["nor60_first_original_60_public_only.json"][1],
        "raw_public_source_bytes": MEMBERS["nor60_first_original_60_public_only.json"][0],
        "raw_first_CPU_sha256": MEMBERS["nor60_first_original_cpu.json"][1],
        "raw_first_CPU_bytes": MEMBERS["nor60_first_original_cpu.json"][0],
        "raw_progress_journal_sha256": MEMBERS["nor60_first_original_progress.jsonl"][1],
        "raw_progress_journal_bytes": MEMBERS["nor60_first_original_progress.jsonl"][0],
        "original_first_five_gates": gates, "first_original_engineering_pass": all(gates.values()),
        "original_primary": primary,
        "original_source_median_wall_ms": cpu["overall"]["source_median_total_ms"],
        "source_count": 60, "exact_vectors_retained": 60, "signed_bundles_retained": 300,
        "primary_timing_samples_retained": 1200,
        "progress_journal_events": cpu["progress_journal_event_count"],
        "scope": "Original finite CPU engineering result under public test keys and an honest external authority premise. No retiming or regenerated evidence."
    }
    receipt_raw = canonical(receipt)
    directory.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw_zip)
    with receipt_path.open("xb") as stream:
        stream.write(receipt_raw)
    return {"archive_sha256": ZIP_SHA, "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "primary": primary, "first_original_engineering_pass": all(gates.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-zip", type=Path, required=True)
    p.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "results")
    args = p.parse_args()
    print(json.dumps(archive_first(args.input_zip, args.directory), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
