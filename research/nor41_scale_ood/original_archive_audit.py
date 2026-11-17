"""Read-only NOR-41 first ORIGINAL source/grade SHA and science-FAIL verifier.

This checker has no new-source constructor and never changes a registered
gate or original receipt. With --full, the separate native, CAL search and
heldout cross-play replayer recomputes all first-grade decisions.
"""
from __future__ import annotations
import argparse, json, sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RESULTS=HERE/"results"
REGISTERED="87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f"
AMENDMENT="e5a95cad1ffdcd019f153cd09dd4d48b40848ef9"
FIRST_RUN=44124795440
SOURCE_FIRST_GENERATION=44124006976
EXPECTED={
    "nor41_fixture.json":(9618649,
       "4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d"),
    "nor41_full.json":(5007355,
       "4efa6bdb1dd3656d0fbb6639065a8f2d934bd4a1acb952c128748cea9918de58")
}
ZIP_BYTES=3388660
ZIP_SHA="0752c8c736ebe4460d02d655d7513662093d5c9dd2ea6e632a0ea76705469d12"
ORIGINAL_GATES={
    "global7_frontier_survives_4096_cal_attack":True,
    "novel_input_family_top_gain_retained":True,
    "scale_sensitive_cal_to_eval_transfer":False,
    "source_first_native_witness_and_full_replay_contract":True,
    "stronger_cal4096_evaluation_transfer":True
}

def audit(full_replay=False):
    manifest=json.loads((RESULTS/"nor41_first_frozen_receipt.json").read_bytes())
    assert manifest["schema"]=="nor41.original_first_nested_cal_ood_archive.v1"
    assert manifest["registration_first_commit"]==REGISTERED
    assert manifest["predata_amendment_01_commit"]==AMENDMENT
    assert manifest["first_source_generation_run"]==SOURCE_FIRST_GENERATION
    assert manifest["first_upload_qualified_grade_run"]==FIRST_RUN
    assert manifest["source_units"]==64
    assert manifest["excluded_prior_source_units"]==324
    assert manifest["calibration_budgets"]==[256,512,2048,4096]
    assert manifest["heldout_schedules"]==1024
    assert manifest["original_scientific_pass"] is False
    assert manifest["original_five_gates"]==ORIGINAL_GATES
    assert manifest["original_file_bytes_sha256"]=={
        key:{"bytes":size,"sha256":sha}
        for key,(size,sha) in EXPECTED.items()}
    assert manifest["source_original_generated_bytes"]==EXPECTED["nor41_fixture.json"][0]
    assert manifest["source_original_generated_sha256"]==EXPECTED["nor41_fixture.json"][1]
    raw=(RESULTS/"nor41_original_first_hosted.zip").read_bytes()
    assert len(raw)==manifest["deterministic_zip_bytes"]==ZIP_BYTES
    assert sha256(raw).hexdigest()==manifest["deterministic_zip_sha256"]==ZIP_SHA
    with ZipFile(RESULTS/"nor41_original_first_hosted.zip") as z:
        assert z.testzip() is None
        assert sorted(z.namelist())==sorted(EXPECTED)
        files={name:z.read(name) for name in EXPECTED}
    for name,(n,h) in EXPECTED.items():
        assert len(files[name])==n and sha256(files[name]).hexdigest()==h
    source=json.loads(files["nor41_fixture.json"])
    grade=json.loads(files["nor41_full.json"])
    assert source["schema"]=="nor41.scale_ood_public_only_reps84to99.v1"
    assert grade["schema"]=="nor41.input_ood_nested_cal_transfer.v1"
    assert source["registered_first_commit"]==grade["registration_first_commit"]==REGISTERED
    assert source["predata_amendment_01_commit"]==grade["predata_amendment_01_commit"]==AMENDMENT
    assert source["public_only"] and source["planted_and_private_target_absent"]
    assert source["attack_witness_and_audit_absent"]
    assert source["excluded_historical_sources"]==grade["historical_sources_excluded"]==324
    assert source["replicates"]==grade["replicates"]==list(range(84,100))
    assert source["degrees"]==grade["degrees"]==[512,1024]
    assert source["families"]==grade["families"]==["wide_dense_small","wide_regime_blocks"]
    assert len(source["cases"])==len(grade["source_records"])==64
    assert len({r["full_public_input_sha256"] for r in source["cases"]})==64
    assert grade["public_fixture_sha256"]==EXPECTED["nor41_fixture.json"][1]
    assert grade["registered_five_gates"]==ORIGINAL_GATES
    assert grade["registered_primary_pass"] is False
    assert len(grade["auditor_policies"])==4
    assert len(grade["blind_control_families"])==6
    assert grade["calibration_prefix_budgets"]==[256,512,2048,4096]
    assert grade["calibration_schedules_per_source"]==4096
    assert grade["heldout_schedules_per_source"]==1024
    assert grade["greedy_starts_per_policy_budget"]==16
    assert grade["selected_attack_sets_per_policy_budget"]==8
    assert grade["exact_pool_combination_count"]==1820
    assert grade["source_means"]==manifest["original_source_means"]
    assert grade["source_direction_wins"]==manifest["original_source_wins"]
    assert all(not set(row).intersection(
        ("program","future","target","attack","witness","audit"))
        for row in source["cases"])
    result={
        "schema":"nor41.original_immutable_source_grade_audit.v1",
        "registration":REGISTERED,
        "amendment":AMENDMENT,
        "source_sha256":EXPECTED["nor41_fixture.json"][1],
        "grade_sha256":EXPECTED["nor41_full.json"][1],
        "zip_sha256":ZIP_SHA,
        "original_scientific_pass":False,
        "registered_five_gates":ORIGINAL_GATES,
        "source_count":64,
        "full_independent_replay":bool(full_replay),
    }
    if full_replay:
        from research.nor41_scale_ood import receipt_audit as independent
        result["replay"]=independent.replay(
            files["nor41_fixture.json"],files["nor41_full.json"])
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit(args.full),sort_keys=True,indent=2))

if __name__=="__main__":
    main()
