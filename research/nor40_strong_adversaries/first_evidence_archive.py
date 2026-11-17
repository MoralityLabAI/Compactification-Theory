"""NOR-40 immutable first-run source/grade publisher, runnable after first grade.

Consumes the ORIGINAL GitHub Actions artifact bytes without regenerating a
source, an audit plan, an attacker, or a graded result. Records scientific
PASS or FAIL exactly as emitted by the frozen scorer, along with all five
registered booleans. Commit the resulting ZIP and JSON manifest unchanged.
"""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile,ZipInfo

FIRST_REGISTRATION="dbc1adbe0e30b9c0e36421cf07cebcf376d6a1ec"
ORIGINAL_RUN_ID=44110593049
ORIGINAL_ARTIFACT="nor40_first_original_source_and_strong_attack_grade"
PUBLIC_FIXTURE_SHA="b51e8031715754c36661b74cff0c26e672fdc21355ff0dc9ab48efded3901de5"
PUBLIC_FIXTURE_BYTES=9387973


def publish(fixture_path:Path,grade_path:Path,destination:Path,artifact_id:int|None):
    original={
        "nor40_fixture.json":fixture_path.read_bytes(),
        "nor40_full.json":grade_path.read_bytes()
    }
    assert len(original["nor40_fixture.json"])==PUBLIC_FIXTURE_BYTES
    assert sha256(original["nor40_fixture.json"]).hexdigest()==PUBLIC_FIXTURE_SHA
    fixture=json.loads(original["nor40_fixture.json"])
    grade=json.loads(original["nor40_full.json"])
    assert fixture["schema"]=="nor40.strong_adversary_public_only_reps76to83.v1"
    assert grade["schema"]=="nor40.strong_calibration_policyaware.v1"
    assert fixture["registered_first_commit"]==grade["registration_first_commit"]==FIRST_REGISTRATION
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["attack_witness_and_audit_absent"]
    assert fixture["excluded_historical_sources"]==grade["historical_sources_excluded"]==276
    assert fixture["replicates"]==list(range(76,84))
    assert fixture["degrees"]==[512,1024]
    assert fixture["source_offset"]==130040000
    assert grade["public_fixture_sha256"]==PUBLIC_FIXTURE_SHA
    assert grade["fresh_source_units"]==len(fixture["cases"])==len(grade["source_records"])==48
    assert len({r["full_public_input_sha256"] for r in fixture["cases"]})==48
    assert all(not set(row)&{"program","future","target","attack","witness","audit"}
               for row in fixture["cases"])
    assert grade["calibration_schedules_per_source"]==2048
    assert grade["baseline_calibration_schedules_per_source"]==256
    assert grade["heldout_schedules_per_source"]==512
    assert grade["greedy_restarts_per_policy"]==32
    assert grade["attack_sets_per_policy"]==8
    assert grade["restricted_pool_size"]==16
    assert grade["restricted_sets_per_policy"]==1820
    assert grade["audit_budget"]==16 and grade["attack_radius"]==4
    assert len(grade["auditor_policies"])==7
    assert len(grade["blind_control_families"])==6
    assert len(grade["registered_five_gates"])==5
    assert type(grade["registered_primary_pass"]) is bool
    assert grade["registered_primary_pass"]==all(
        bool(v) for v in grade["registered_five_gates"].values())
    assert all(len(rec["attack_certificates"])==7
               and len(rec["greedy_attack_sets"])==7
               and len(rec["baseline_attack_sets"])==7
               and len(rec["blind_attack_sets"])==6
               and len(rec["crossplay_weighted_numerators"])==20
               and all(cert["restricted_sets_enumerated"]==1820
                       for cert in rec["attack_certificates"].values())
               for rec in grade["source_records"])
    assert grade["public_proposals_frozen_before_private_truth"]
    assert grade["calibrated_attacks_frozen_before_independent_heldout_schedules"]
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/"nor40_original_first_hosted.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED,compresslevel=9) as zipped:
        for name,data in original.items():
            info=ZipInfo(name,(2026,11,17,0,0,0))
            info.compress_type=ZIP_DEFLATED
            zipped.writestr(info,data,compress_type=ZIP_DEFLATED,compresslevel=9)
    zipbytes=archive.read_bytes()
    hashes={name:{"bytes":len(data),"sha256":sha256(data).hexdigest()}
            for name,data in original.items()}
    manifest={
        "schema":"nor40.first_original_strong_adversary_archive.v1",
        "first_registration_commit":FIRST_REGISTRATION,
        "first_original_grade_run":ORIGINAL_RUN_ID,
        "first_original_artifact_name":ORIGINAL_ARTIFACT,
        "first_original_artifact_id":artifact_id,
        "source_only_first_fixture_frozen_before_grade":True,
        "source_only_first_fixture_sha256":PUBLIC_FIXTURE_SHA,
        "original_file_bytes_sha256":hashes,
        "archive_zip_bytes":len(zipbytes),
        "archive_zip_sha256":sha256(zipbytes).hexdigest(),
        "historical_source_units_excluded":276,
        "new_source_units":48,
        "calibration_schedules_per_source":2048,
        "baseline_calibration_schedules_per_source":256,
        "heldout_schedules_per_source":512,
        "greedy_starts_per_policy":32,
        "selected_attack_sets_per_policy":8,
        "restricted_search_pool_size":16,
        "restricted_sets_per_policy":1820,
        "source_means":grade["source_means"],
        "source_direction_wins":grade["source_direction_wins"],
        "original_five_gates":grade["registered_five_gates"],
        "original_science_pass":grade["registered_primary_pass"],
        "limitation":"Partial-support threat model uses authenticated true-target integer coefficient witness and genuinely hidden audit randomness. Reproducible SHA streams are public fixtures. Pool16 exact enumeration is not full-domain proof."
    }
    receipt=destination/"nor40_first_frozen_receipt.json"
    receipt.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("NOR40 ORIGINAL SOURCE SHA",PUBLIC_FIXTURE_SHA)
    print("NOR40 ORIGINAL GRADE BYTES/SHA",hashes["nor40_full.json"])
    print("NOR40 ZIP BYTES/SHA",manifest["archive_zip_bytes"],manifest["archive_zip_sha256"])
    print("NOR40 ORIGINAL FIVE GATES",grade["registered_five_gates"])
    print("NOR40 SCIENTIFIC PASS",grade["registered_primary_pass"])
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",required=True,type=Path)
    p.add_argument("--grade",required=True,type=Path)
    p.add_argument("--destination",required=True,type=Path)
    p.add_argument("--artifact-id",type=int)
    args=p.parse_args()
    publish(args.fixture,args.grade,args.destination,args.artifact_id)


if __name__=="__main__":
    main()
