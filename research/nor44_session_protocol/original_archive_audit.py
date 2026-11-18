"""NOR-44 immutable first historical signed-session raw receipt audit.

No new sources, signatures, nonces or challenges are regenerated here.
Original first 64-source session data stay byte-for-byte unchanged.
"""
from __future__ import annotations

import json
import statistics
from hashlib import sha256
from pathlib import Path

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/"results/nor44_first_historical_64_sessions.json"
FIRST_BYTES=121592
FIRST_SHA="b7f740b6916e651932dbd40ceee443bdd5fb40c4b7a7042830ef53f749bd1f83"
NOR42_SOURCE="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
NOR42_GRADE="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
NOR43_FIRST="f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145"


def verify():
    raw=ORIGINAL.read_bytes()
    assert len(raw)==FIRST_BYTES and sha256(raw).hexdigest()==FIRST_SHA
    x=json.loads(raw)
    assert x["schema"]=="nor44.archived_nor42_session_ordering_smoke.v1"
    assert x["original_nor42_source_sha256"]==NOR42_SOURCE
    assert x["original_nor42_grade_sha256"]==NOR42_GRADE
    assert x["prior_nor43_original_smoke_sha256"]==NOR43_FIRST
    assert x["historical_source_pair_count"]==64
    assert x["new_nor44_prospective_source_count"]==0
    assert x["possible_source_truth_worlds"]==128
    assert x["signed_sessions_per_source"]==3
    assert x["challenge_queries_per_session"]==16
    assert x["first_model_proposals_committed_before_reference"]
    assert x["reference_roots_signed_under_pinned_test_provider"]
    assert x["challenge_indices_bound_to_verifier_state"]
    assert x["all_valid_proof_substitution_attacks_rejected"]
    assert x["all_dishonest_authorized_provider_cases_crypto_accept_false_truth"]
    assert x["test_signing_keys_are_public"]
    assert x["test_nonces_and_challenges_are_public_deterministic"]
    rows=x["source_records"]
    assert len(rows)==len({r["nor42_public_source_sha256"] for r in rows})==64
    assert {r["degree"] for r in rows}=={512,1024}
    assert {r["family"] for r in rows}=={
        "wide_dense_small","wide_regime_blocks"}
    assert {r["replicate"] for r in rows}==set(range(100,116))
    for rep in range(100,116):
        assert sum(r["replicate"]==rep for r in rows)==4
    for row in rows:
        assert row["test_only_randomness"] is True
        assert row["source_output_length"]==3*row["degree"]//2+1
        assert row["challenge_queries_per_session"]==16 if (
            "challenge_queries_per_session" in row) else True
        assert len(row["challenge_indices_sorted"])==16
        assert len(set(row["challenge_indices_sorted"]))==16
        assert all(0<=i<row["source_output_length"] for i in row["challenge_indices_sorted"])
        assert row["valid_proof_substitution_rejected"]
        assert row["dishonest_authorized_provider_signs_false_vector_but_passes_crypto"]
        assert row["dishonest_authorized_provider_reference_is_true"] is False
        assert row["challenge_rejected_wrong_clean_proposal"]==row[
            "challenge_intersects_true_difference"]
        for field in (
           "session_nonce_hex","source_session_proposal_sha256",
           "signed_verifier_precommit_sha256","signed_provider_reference_sha256",
           "signed_challenge_sha256","signed_terminal_sha256"):
            value=row[field]
            assert len(value)==64 and all(c in "0123456789abcdef" for c in value)
        assert len(row["uniform16_exact_detection_fraction"])==2
        assert row["signed_verifier_precommit_bytes"]>64
        assert row["signed_provider_reference_bytes"]>64
        assert row["signed_challenge_bytes"]>64
        assert row["signed_terminal_bytes"]>64
        assert row["sixteen_original_merkle_openings_bytes"]>1000
    families={}
    for name in ("wide_dense_small","wide_regime_blocks"):
        group=[row for row in rows if row["family"]==name]
        actual=x["families"][name]
        assert actual["source_pair_count"]==len(group)==32
        rate=statistics.mean(row["challenge_rejected_wrong_clean_proposal"]
                             for row in group)
        assert actual["fixed_public_test_k16_detection_rate"]==rate
        assert actual["mean_merkle_depth"]==statistics.mean(
            row["merkle_depth"] for row in group)
        for k in ("signed_verifier_precommit_bytes","signed_provider_reference_bytes",
                  "signed_challenge_bytes","signed_terminal_bytes",
                  "sixteen_original_merkle_openings_bytes"):
            assert actual["mean_"+k]==statistics.mean(row[k] for row in group)
        families[name]={"pairs":len(group),"fixed_test_hits":sum(
            row["challenge_rejected_wrong_clean_proposal"] for row in group),
            "mean_k16_proof_bytes":actual["mean_sixteen_original_merkle_openings_bytes"],
            "mean_precommit_receipt_bytes":actual["mean_signed_verifier_precommit_bytes"],
            "mean_reference_bytes":actual["mean_signed_provider_reference_bytes"],
            "mean_challenge_bytes":actual["mean_signed_challenge_bytes"],
            "mean_terminal_bytes":actual["mean_signed_terminal_bytes"]}
    return {
        "schema":"nor44.original_historical_session_receipt_readonly_audit.v1",
        "original_first_run":44228006426,
        "original_first_sha256":FIRST_SHA,
        "original_first_bytes":FIRST_BYTES,
        "historical_source_pair_units":64,
        "fresh_nor44_source_units":0,
        "challenge_substitution_attack_rejections":64,
        "authorized_but_dishonest_false_root_controls":64,
        "deterministic_insecure_fixture":True,
        "families":families
    }


if __name__=="__main__":
    print(json.dumps(verify(),indent=2,sort_keys=True))
