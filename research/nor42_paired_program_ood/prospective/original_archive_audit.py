"""NOR-42 permanent first ORIGINAL source, grade and archive SHA verifier.

Read-only. Source-only public fixture, first original full grade, and
first registered five science gates remain unchanged. --full optionally
runs the separate complete native program/hidden world/plan replayer.
"""
from __future__ import annotations
import argparse,json,sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RESULTS=HERE/"results"
REGISTERED="9bf353b7c76535e581ff026098cd682914a90ef0"
FIRST_RUN=44138400655
FIRST_ARTIFACT=14352479533
FIRST={
    "nor42_fixture.json":(9059929,
      "f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"),
    "nor42_full.json":(201956,
      "bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae")
}
ZIP_BYTES=2799867
ZIP_SHA="e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d"
GATES={
    "paired_prefix_identity_and_future_underdetermination":True,
    "authenticated_uniform_exact_support_calibration":True,
    "authenticated_hidden_detection_over_witnessfree_consistency":True,
    "global7_competitive_offgrammar_new_family":True,
    "source_first_hidden_witness_and_independent_replay_contract":True
}


def check(full=False):
    receipt=json.loads((RESULTS/"nor42_first_frozen_receipt.json").read_bytes())
    assert receipt["schema"]=="nor42.original_first_paired_grammar_ood_archive.v1"
    assert receipt["first_registration_commit"]==REGISTERED
    assert receipt["original_first_run"]==FIRST_RUN
    assert receipt["first_original_grade_artifact_id"]==FIRST_ARTIFACT
    assert receipt["source_only_first_fixture_frozen_before_true_worlds"]
    assert receipt["new_source_pair_units"]==64
    assert receipt["paired_true_worlds"]==128
    assert receipt["excluded_historical_source_ids"]==388
    assert receipt["heldout_schedules_per_source"]==1024
    assert receipt["audit_queries_per_plan"]==16
    assert receipt["original_registered_science_pass"] is True
    assert receipt["original_five_gates"]==GATES
    assert receipt["original_raw_file_bytes_sha256"]=={
        name:{"bytes":n,"sha256":h} for name,(n,h) in FIRST.items()}
    zipraw=(RESULTS/"nor42_original_first_hosted.zip").read_bytes()
    assert len(zipraw)==receipt["deterministic_zip_bytes"]==ZIP_BYTES
    assert sha256(zipraw).hexdigest()==receipt["deterministic_zip_sha256"]==ZIP_SHA
    with ZipFile(RESULTS/"nor42_original_first_hosted.zip") as z:
        assert z.testzip() is None
        assert sorted(z.namelist())==sorted(FIRST)
        originals={name:z.read(name) for name in FIRST}
    for name,(size,h) in FIRST.items():
        assert len(originals[name])==size
        assert sha256(originals[name]).hexdigest()==h
    source=json.loads(originals["nor42_fixture.json"])
    grade=json.loads(originals["nor42_full.json"])
    assert source["schema"]=="nor42.paired_grammar_ood_public_only_reps100to115.v1"
    assert grade["schema"]=="nor42.paired_program_ood_authenticated_witness.v1"
    assert source["registered_first_commit"]==grade["registration_first_commit"]==REGISTERED
    assert source["public_only"] and source["planted_and_private_target_absent"]
    assert source["hidden_beta_transform_absent"] and source["attack_witness_and_audit_absent"]
    assert source["source_units"]==grade["fresh_public_source_pairs"]==64
    assert source["private_truth_worlds"]==grade["two_world_targets"]==128
    assert source["excluded_historical_sources"]==grade["excluded_historical_sources"]==388
    assert source["replicates"]==list(range(100,116))
    assert source["degrees"]==[512,1024]
    assert source["families"]==["wide_dense_small","wide_regime_blocks"]
    assert grade["public_fixture_sha256"]==FIRST["nor42_fixture.json"][1]
    ids=[r["full_public_input_sha256"] for r in source["cases"]]
    assert len(ids)==len(set(ids))==64
    assert len(grade["source_records"])==64
    assert len(grade["source_level_primary"])==64
    assert grade["auditor_policies"]==[
        "uniform16","original_mix","deterministic_mix","global7_mix"]
    assert grade["diagnostic_auditor"]=="atom_random_h50"
    assert grade["audit_budget"]==16
    assert grade["heldout_schedules_per_source"]==1024
    assert grade["hidden_suffix_offset_after_public"]==32
    assert grade["hidden_suffix_salt"]==202611420091
    assert grade["registered_five_gates"]==GATES
    assert grade["registered_primary_pass"] is True
    assert grade["source_means"]==receipt["original_means"]
    assert grade["source_direction_wins"]==receipt["original_wins"]
    assert all(r["paired_world_labels"]==["clean","hidden"]
               and r["paired_world_prior"]==[1,2]
               and not set(r).intersection(
                   ("program","future","target","attack","witness","audit","beta"))
               for r in source["cases"])
    from research.nor42_paired_program_ood.prospective import receipt_audit as independent
    previous=independent.history()
    assert len(previous)==388
    assert set(ids).isdisjoint(previous)
    result={
        "schema":"nor42.first_original_immutable_archive_audit.v1",
        "registered_first_commit":REGISTERED,
        "source_sha256":FIRST["nor42_fixture.json"][1],
        "grade_sha256":FIRST["nor42_full.json"][1],
        "zip_sha256":ZIP_SHA,
        "source_pairs":64,"previous_shas_excluded":388,
        "registered_primary_pass":True,
        "five_original_gates":GATES,
        "independent_full_replay_requested":bool(full)
    }
    if full:
        result["independent"]=independent.replay(
            originals["nor42_fixture.json"],originals["nor42_full.json"])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    print(json.dumps(check(args.full),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
