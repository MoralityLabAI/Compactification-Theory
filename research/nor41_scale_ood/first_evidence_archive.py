"""NOR-41 original first source+grade immutable archival writer.

Runs only after the scorer has completed its first original heldout evaluation.
Never regenerates any source, audit plan, target program or attack. Preserves
a prospective science FAIL exactly as the first grader emitted it.
"""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile,ZipInfo

REGISTERED="87dd3de03f43e79d83d45ed7a5c3a3bb1f20746f"
AMENDMENT="e5a95cad1ffdcd019f153cd09dd4d48b40848ef9"
SOURCE_SHA="4073090e5888ed10283d82dc5153758d299ed7cfe9c753be39ba2c2bd4c7119d"
SOURCE_LENGTH=9618649
FIRST_SOURCE_GENERATION_RUN=44124006976


def archive(fixture:Path,grade:Path,output:Path,first_grade_run:int,
            grade_artifact_id:int|None):
    original={"nor41_fixture.json":fixture.read_bytes(),
              "nor41_full.json":grade.read_bytes()}
    assert len(original["nor41_fixture.json"])==SOURCE_LENGTH
    assert sha256(original["nor41_fixture.json"]).hexdigest()==SOURCE_SHA
    f=json.loads(original["nor41_fixture.json"])
    g=json.loads(original["nor41_full.json"])
    assert f["schema"]=="nor41.scale_ood_public_only_reps84to99.v1"
    assert g["schema"]=="nor41.input_ood_nested_cal_transfer.v1"
    assert f["registered_first_commit"]==g["registration_first_commit"]==REGISTERED
    assert f["predata_amendment_01_commit"]==g["predata_amendment_01_commit"]==AMENDMENT
    assert f["public_only"] and f["planted_and_private_target_absent"]
    assert f["attack_witness_and_audit_absent"]
    assert f["excluded_historical_sources"]==g["historical_sources_excluded"]==324
    assert g["public_fixture_sha256"]==SOURCE_SHA
    assert f["replicates"]==g["replicates"]==list(range(84,100))
    assert f["degrees"]==g["degrees"]==[512,1024]
    assert f["families"]==g["families"]==[
        "wide_dense_small","wide_regime_blocks"]
    assert len(f["cases"])==len(g["source_records"])==g["fresh_source_units"]==64
    assert len({r["full_public_input_sha256"] for r in f["cases"]})==64
    assert g["calibration_prefix_budgets"]==[256,512,2048,4096]
    assert g["calibration_schedules_per_source"]==4096
    assert g["heldout_schedules_per_source"]==1024
    assert g["greedy_starts_per_policy_budget"]==16
    assert g["selected_attack_sets_per_policy_budget"]==8
    assert g["exact_candidate_pool_size"]==16
    assert g["exact_pool_combination_count"]==1820
    assert len(g["auditor_policies"])==4
    assert len(g["blind_control_families"])==6
    assert len(g["registered_five_gates"])==5
    assert type(g["registered_primary_pass"]) is bool
    assert all(type(value) is bool for value in g["registered_five_gates"].values())
    assert g["registered_primary_pass"]==all(g["registered_five_gates"].values())
    assert all(len(rec["calibrated_attacks"])==4
               and len(rec["attack_certificates"])==4
               and len(rec["crossplay_weighted_numerators"])==22
               and all(len(rec["calibrated_attacks"][str(n)])==4
                       and all(len(rec["attack_certificates"][str(n)])==4
                               for n in (256,512,2048,4096))
               for rec in g["source_records"])
    output.mkdir(parents=True,exist_ok=True)
    archive_file=output/"nor41_original_first_hosted.zip"
    with ZipFile(archive_file,"w",ZIP_DEFLATED,compresslevel=9) as z:
        for name,bytes_ in original.items():
            info=ZipInfo(name,(2026,11,17,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,bytes_,compress_type=ZIP_DEFLATED,compresslevel=9)
    zip_bytes=archive_file.read_bytes()
    sha={name:{"bytes":len(data),"sha256":sha256(data).hexdigest()}
         for name,data in original.items()}
    manifest={
        "schema":"nor41.original_first_nested_cal_ood_archive.v1",
        "registration_first_commit":REGISTERED,
        "predata_amendment_01_commit":AMENDMENT,
        "first_source_generation_run":FIRST_SOURCE_GENERATION_RUN,
        "first_source_generation_check_failure":
           "Prior 48-source count assertion after successful 64-source generation; "
           "no CAL, EVAL, private target or grade executed in that stopped run",
        "first_upload_qualified_grade_run":first_grade_run,
        "first_original_grade_artifact_id":grade_artifact_id,
        "source_original_generated_bytes":SOURCE_LENGTH,
        "source_original_generated_sha256":SOURCE_SHA,
        "original_file_bytes_sha256":sha,
        "deterministic_zip_bytes":len(zip_bytes),
        "deterministic_zip_sha256":sha256(zip_bytes).hexdigest(),
        "source_units":64,"excluded_prior_source_units":324,
        "calibration_budgets":[256,512,2048,4096],
        "heldout_schedules":1024,
        "original_source_means":g["source_means"],
        "original_source_wins":g["source_direction_wins"],
        "original_five_gates":g["registered_five_gates"],
        "original_scientific_pass":g["registered_primary_pass"],
        "separate_independent_replay":"Track GitHub Actions original run independently",
        "scope":"Input-family OOD only; original program grammar. The source-first "
                 "fixture contains public inputs and observed prefixes only. "
                 "Authenticated output witness and hidden coins are external."
    }
    receipt=output/"nor41_first_frozen_receipt.json"
    receipt.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("NOR41 FIRST SOURCE ORIGINAL SHA",SOURCE_SHA)
    print("NOR41 FIRST GRADE BYTES SHA",sha["nor41_full.json"])
    print("NOR41 ORIGINAL ZIP BYTES SHA",
          manifest["deterministic_zip_bytes"],manifest["deterministic_zip_sha256"])
    print("NOR41 ORIGINAL SCIENCE PASS",manifest["original_scientific_pass"])
    print("NOR41 ORIGINAL GATES",manifest["original_five_gates"])
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",required=True,type=Path)
    p.add_argument("--grade",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--first-grade-run",required=True,type=int)
    p.add_argument("--artifact-id",type=int)
    a=p.parse_args()
    archive(a.fixture,a.grade,a.out,a.first_grade_run,a.artifact_id)


if __name__=="__main__":
    main()
