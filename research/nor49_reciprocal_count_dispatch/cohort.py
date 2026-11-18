"""NOR49 timing-clean source ID freeze before any first 48-source CPU run.

Historical original NOR42 inputs are public and previously source-frozen.
NOR46–48 timing TRAIN 16 identities are excluded by immutable first SHA.
No heldout execution, benchmark timing, true future, or signed program op
selection is performed by this roster-only process.
"""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor48_program_dispatch import benchmark_cpu as nor48

TRAIN_REPS=(100,104,108,112)
TEST_REPS=(101,102,103,105,106,107,109,110,111,113,114,115)
REGISTERED="26b70716ec3d3830031919e593d3092cc31ae162"
ORIGINAL_NOR42_SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
ORIGINAL_NOR48_FIRST_CPU_SHA="d5cf4b0e3860907eb58c2a37b84ff631f023caf2b69e6e0d0e5a322e2f0b2b99"


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def freeze_roster():
    training=nor48.source_provenance()
    original_fixture,grade=nor48.old.nor45.load_original()
    assert original_fixture["schema"]=="nor42.paired_grammar_ood_public_only_reps100to115.v1"
    allids=[r["full_public_input_sha256"] for r in original_fixture["cases"]]
    assert len(allids)==len(set(allids))==64
    training_ids=set(training)
    assert len(training_ids)==16
    seen=[r for r in original_fixture["cases"] if r["full_public_input_sha256"] in training_ids]
    assert len(seen)==16
    assert Counter(r["pair_replicate"] for r in seen)=={rep:4 for rep in TRAIN_REPS}
    rows=[]
    for r in original_fixture["cases"]:
        if r["full_public_input_sha256"] in training_ids:continue
        assert r["pair_replicate"] in TEST_REPS
        rows.append({
            "source_sha256":r["full_public_input_sha256"],
            "replicate":r["pair_replicate"],
            "degree":r["degree"],
            "family":r["family"],
            "public_input_length":len(r["public_query_input"]),
            "public_mask_cardinality":r["original_native_public_mask"].bit_count()
        })
    assert len(rows)==len({r["source_sha256"] for r in rows})==48
    assert Counter(r["replicate"] for r in rows)=={rep:4 for rep in TEST_REPS}
    assert Counter(r["degree"] for r in rows)=={512:24,1024:24}
    assert Counter(r["family"] for r in rows)=={
        "wide_dense_small":24,"wide_regime_blocks":24}
    return {
        "schema":"nor49.first_fixed_timing_clean_nor42_source_roster.v1",
        "first_registration_sha":REGISTERED,
        "nor42_first_public_source_sha256":ORIGINAL_NOR42_SOURCE_SHA,
        "nor48_first_cpu_train_sha256":ORIGINAL_NOR48_FIRST_CPU_SHA,
        "training_source_count":16,
        "timing_clean_source_count":48,
        "excluded_training_replicates":list(TRAIN_REPS),
        "selected_test_replicates":list(TEST_REPS),
        "no_new_source_identity_created":True,
        "no_holdout_timing_observed":True,
        "selection_contains_no_program_or_private_truth":True,
        "cases":rows
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    raw=canonical(freeze_roster())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(raw)
    print("NOR49 FIRST TIMING-CLEAN SOURCE ROSTER BYTES SHA",len(raw),sha256(raw).hexdigest())


if __name__=="__main__":
    main()
