"""NOR-27 permanent SHA-pinned first-run evidence auditor.

The source-controlled original ZIP retains the exact original prospective
source-only fixture and full result bytes. This script never imports NOR27's
experiment or replayer. GitHub Actions checks the historical original bytes
without overwriting or regenerating them.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
RECEIPT=HERE/"results"/"nor27_first_original_receipt.json"
ARCHIVE=HERE/"results"/"nor27_original_hosted.zip"


def check():
    expected=json.loads(RECEIPT.read_text())
    assert expected["schema"]=="nor27.first_original_hosted_sha_locked_receipt.v1"
    assert expected["registered_first_commit"]=="bcc49dfb96acd9a9aae51925630b2f04f69e2315"
    assert expected["first_hosted_run_id"]==44015771328
    assert expected["first_original_artifact_id"]==14298756859
    assert ARCHIVE.is_file(),"original NOR27 first CI ZIP not yet committed"
    data={}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert set(z.namelist())=={
            expected["fixture"]["name"],expected["outcome"]["name"]}
        for key in ("fixture","outcome"):
            meta=expected[key]
            original=z.read(meta["name"])
            assert len(original)==meta["bytes"],(key,len(original),meta["bytes"])
            assert sha256(original).hexdigest()==meta["sha256"],(key,"SHA256")
            data[key]=json.loads(original)
    source=data["fixture"]
    full=data["outcome"]
    assert source["source_only"] and source["hidden_truth_absent"]
    assert source["audit_seeds_absent"] and source["leakage_and_attack_absent"]
    assert len(source["cases"])==12
    assert len({x["full_public_input_sha256"] for x in source["cases"]})==12
    assert full["schema"]=="nor27.seed_prefix_and_reserve_preview_leakage.v1"
    assert full["source_count"]==12 and len(full["source_records"])==12
    assert full["fixture_sha256"]==expected["fixture"]["sha256"]
    assert full["registered_primary_pass"]==expected["registered_scientific_pass"]
    for key,gate in (
        ("primary_preview_degradation_pass","preview_degradation"),
        ("primary_preview_theory_agreement_pass","preview_degree_agreement"),
        ("primary_seed_prefix_sensitivity_pass","raw_seed_bits_sensitivity"),
        ("primary_exact_witness_controls_pass","exact_witness_controls")):
        assert full[key]==expected["primary_gates"][gate]
    assert full["preview_source_means"]=={
        str(k):v for k,v in expected["preview_source_means"].items()}
    assert full["seed_prefix_source_means"]==expected["seed_prefix_source_means"]
    assert full["preview_empirical_by_degree"]==expected["preview_empirical_by_output_length"]
    assert full["preview_theoretical_by_degree"]==expected["preview_theoretical_by_output_length"]
    assert all(len(r["preview"])==5 and len(r["seed_prefix_buckets"])==42
               and len(r["seed_summary"])==6 and
               r["honest_target_failures"]==0 and
               r["delta_disagreements"]==0 and
               r["fully_informed_preview_detections"]==0
               for r in full["source_records"])
    print("NOR27 permanent first original SHA-locked audit PASS")
    print("original public-only SHA256",expected["fixture"]["sha256"])
    print("original prospective outcome SHA256",expected["outcome"]["sha256"])


if __name__=="__main__":
    check()
