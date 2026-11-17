"""NOR-42 first ORIGINAL source and paired-world grade SHA archive writer.

Read-only over the first hosted fixture and grade: preserves the exact first
registered PASS or FAIL without regenerating a source, program, audit schedule
or result. Invoked only after the original source-first grader completes.
"""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile,ZipInfo

FIRST_REGISTRATION="9bf353b7c76535e581ff026098cd682914a90ef0"
FIRST_RUN_ID=44138400655
FIRST_SOURCE_BYTES=9059929
FIRST_SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"


def preserve(fixture_path:Path,grade_path:Path,out:Path,artifact_id:int|None):
    originals={
        "nor42_fixture.json":fixture_path.read_bytes(),
        "nor42_full.json":grade_path.read_bytes()
    }
    src=originals["nor42_fixture.json"]
    graded=originals["nor42_full.json"]
    assert len(src)==FIRST_SOURCE_BYTES
    assert sha256(src).hexdigest()==FIRST_SOURCE_SHA
    fixture=json.loads(src)
    grade=json.loads(graded)
    assert fixture["schema"]=="nor42.paired_grammar_ood_public_only_reps100to115.v1"
    assert grade["schema"]=="nor42.paired_program_ood_authenticated_witness.v1"
    assert fixture["registered_first_commit"]==grade["registration_first_commit"]==FIRST_REGISTRATION
    assert fixture["source_units"]==grade["fresh_public_source_pairs"]==64
    assert fixture["private_truth_worlds"]==grade["two_world_targets"]==128
    assert fixture["excluded_historical_sources"]==grade["excluded_historical_sources"]==388
    assert grade["public_fixture_sha256"]==FIRST_SOURCE_SHA
    assert fixture["public_only"] and fixture["planted_and_private_target_absent"]
    assert fixture["hidden_beta_transform_absent"] and fixture["attack_witness_and_audit_absent"]
    assert fixture["replicates"]==list(range(100,116))
    assert fixture["degrees"]==[512,1024]
    assert fixture["families"]==["wide_dense_small","wide_regime_blocks"]
    assert len(fixture["cases"])==len(grade["source_records"])==64
    assert len({r["full_public_input_sha256"] for r in fixture["cases"]})==64
    assert grade["audit_budget"]==16 and grade["heldout_schedules_per_source"]==1024
    assert grade["hidden_suffix_offset_after_public"]==32
    assert grade["hidden_suffix_salt"]==202611420091
    assert grade["auditor_policies"]==[
        "uniform16","original_mix","deterministic_mix","global7_mix"]
    assert grade["diagnostic_auditor"]=="atom_random_h50"
    assert len(grade["registered_five_gates"])==5
    assert type(grade["registered_primary_pass"]) is bool
    assert all(type(v) is bool for v in grade["registered_five_gates"].values())
    assert grade["registered_primary_pass"]==all(grade["registered_five_gates"].values())
    assert all(rec["paired_worlds"]==2
               and rec["public_predictor_correct_worlds"]==1
               and rec["first_changed_index"]==rec["degree"]+32
               and rec["first_hidden_coefficient_delta"]==rec["hidden_beta"]!=0
               and len(rec["mixture_detection_numerators"])==5
               and rec["mixture_detection_denominator"]==256*1024
               for rec in grade["source_records"])
    out.mkdir(parents=True,exist_ok=True)
    archive=out/"nor42_original_first_hosted.zip"
    with ZipFile(archive,"w",ZIP_DEFLATED,compresslevel=9) as z:
        for name,buf in originals.items():
            entry=ZipInfo(name,(2026,11,18,0,0,0))
            entry.compress_type=ZIP_DEFLATED
            z.writestr(entry,buf,compress_type=ZIP_DEFLATED,compresslevel=9)
    raw=archive.read_bytes()
    hashes={key:{"bytes":len(buf),"sha256":sha256(buf).hexdigest()}
            for key,buf in originals.items()}
    result={
        "schema":"nor42.original_first_paired_grammar_ood_archive.v1",
        "first_registration_commit":FIRST_REGISTRATION,
        "original_first_run":FIRST_RUN_ID,
        "first_original_grade_artifact_id":artifact_id,
        "original_first_source_bytes":FIRST_SOURCE_BYTES,
        "original_first_source_sha256":FIRST_SOURCE_SHA,
        "original_raw_file_bytes_sha256":hashes,
        "deterministic_zip_bytes":len(raw),
        "deterministic_zip_sha256":sha256(raw).hexdigest(),
        "source_only_first_fixture_frozen_before_true_worlds":True,
        "new_source_pair_units":64,
        "paired_true_worlds":128,
        "excluded_historical_source_ids":388,
        "heldout_schedules_per_source":1024,
        "audit_queries_per_plan":16,
        "original_means":grade["source_means"],
        "original_wins":grade["source_direction_wins"],
        "original_five_gates":grade["registered_five_gates"],
        "original_registered_science_pass":grade["registered_primary_pass"],
        "external_witness_premise":"Synthetic oracle for true clean/hidden output coefficients; no real-world authenticated witness constructed.",
        "independent_replay":"Original separate CI replayer must be verified externally"
    }
    receipt=out/"nor42_first_frozen_receipt.json"
    receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NOR42 ORIGINAL FIRST SOURCE BYTES SHA256",
          len(src),hashes["nor42_fixture.json"]["sha256"])
    print("NOR42 ORIGINAL FIRST GRADE BYTES SHA256",
          len(graded),hashes["nor42_full.json"]["sha256"])
    print("NOR42 FIRST REGISTERED SCIENTIFIC PASS",grade["registered_primary_pass"])
    print("NOR42 FIVE REGISTERED GATES",grade["registered_five_gates"])
    print("NOR42 ORIGINAL ZIP BYTES SHA256",len(raw),result["deterministic_zip_sha256"])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",required=True,type=Path)
    parser.add_argument("--grade",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--artifact-id",type=int)
    args=parser.parse_args()
    preserve(args.fixture,args.grade,args.out,args.artifact_id)


if __name__=="__main__":
    main()
