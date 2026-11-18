"""Read-only NOR45 ORIGINAL first historical signed-program receipt audit.

Checks the 112,573-byte SHA-locked JSON from the first successfully hosted
64-pair historical smoke. No new source generation, keys, targets, plans or
verifier state. Recomputes source counts, evidence and all exact support
probabilities with a separately written finite combinatorial formula.
"""
from __future__ import annotations
import json,statistics
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIRST=HERE/"results/nor45_first_archived_64_programs.json"
BYTES=112573
SHA="c4b1197b87b43c4cb4e32e0c2708c4a070742244d7cc05639e22353af78d5536"
SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
GRADE_SHA="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
FIELDS=(
    "all_authorized_program_executions_match_signed_hidden_reference",
    "all_dishonest_signed_output_roots_rejected_by_program_recomputation",
    "all_model_self_signed_programs_rejected",
    "all_dishonest_authorized_programs_pass_crypto_but_miss_true_world")


def audit():
    raw=FIRST.read_bytes()
    assert len(raw)==BYTES
    assert sha256(raw).hexdigest()==SHA
    data=json.loads(raw)
    assert data["schema"]=="nor45.authorized_program_historical_nor42_64_pairs.v1"
    assert data["original_nor42_source_sha256"]==SOURCE_SHA
    assert data["original_nor42_grade_sha256"]==GRADE_SHA
    assert data["historical_original_source_pairs"]==64
    assert data["new_nor45_prospective_source_ids"]==0
    assert data["program_authority_private_key_public_test_fixture"] is True
    assert data["verifier_test_challenge_predictable"] is True
    assert all(data[field] is True for field in FIELDS)
    rows=data["source_records"]
    assert len(rows)==len({x["original_nor42_source_sha256"] for x in rows})==64
    assert Counter(x["family"] for x in rows)=={
        "wide_dense_small":32,"wide_regime_blocks":32}
    assert Counter(x["degree"] for x in rows)=={512:32,1024:32}
    assert Counter(x["replicate"] for x in rows)=={
        rep:4 for rep in range(100,116)}
    controls=(
        "precommit_source_and_session_bound",
        "independent_program_reexecution_matches_hidden_true_world",
        "independently_rebuilt_hidden_root_matches_publisher",
        "unauthorized_model_program_rejected",
        "dishonest_authorized_output_provider_rejected",
        "dishonest_authorized_program_semantic_failure",
        "all_test_private_keys_publicly_known",
        "test_challenge_is_publicly_reproducible",
        "session_bound_to_original_source")
    for record in rows:
        assert all(record[c] is True for c in controls)
        d=record["degree"]
        m=record["future_coefficients"]
        assert m==3*d//2+1
        s=record["original_hidden_difference_support_size"]
        assert 1<=s<=m
        checked=record["challenge_indices"]
        assert len(checked)==len(set(checked))==16
        assert all(type(index) is int and 0<=index<m for index in checked)
        exact=(Fraction(1) if m-s<16 else
               Fraction(1)-Fraction(comb(m-s,16),comb(m,16)))
        assert record["exact_hidden_uniform16_probability_fraction"]==[
            exact.numerator,exact.denominator]
        assert record["signed_program_certificate_bytes"]>500
        assert record["sixteen_merkle_openings_bytes"]>1000
        assert record["provider_signed_reference_bytes"]>200
        assert record["signed_terminal_receipt_bytes"]>200
    summary={}
    for family in ("wide_dense_small","wide_regime_blocks"):
        group=[x for x in rows if x["family"]==family]
        x=data["family_summaries"][family]
        assert x["n"]==32
        assert x["sampled_k16_detection"]==sum(
            r["authenticated_challenge_rejects_clean_proposal"] for r in group)
        expectation=statistics.mean(
            float(Fraction(*r["exact_hidden_uniform16_probability_fraction"]))
            for r in group)
        assert isclose(x["exact_k16_detection_mean"],expectation,abs_tol=1e-15)
        cert_bytes=statistics.mean(r["signed_program_certificate_bytes"] for r in group)
        proof_bytes=statistics.mean(r["sixteen_merkle_openings_bytes"] for r in group)
        assert isclose(x["mean_program_cert_bytes"],cert_bytes,abs_tol=1e-12)
        assert isclose(x["mean_proof_bytes"],proof_bytes,abs_tol=1e-12)
        summary[family]={"n":32,"source_mean_exact_hit":expectation,
                         "sampled_hit_count":x["sampled_k16_detection"],
                         "mean_program_certificate_bytes":cert_bytes,
                         "mean_sixteen_proof_bytes":proof_bytes}
    assert sum(x["sampled_hit_count"] for x in summary.values())==60
    return {
        "schema":"nor45.first_original_archived_signed_program_audit.v1",
        "original_bytes":BYTES,
        "original_sha256":SHA,
        "previous_nor42_public_sha256":SOURCE_SHA,
        "previous_nor42_grade_sha256":GRADE_SHA,
        "source_pairs":64,
        "new_prospective_source_ids":0,
        "program_authority_integrity_pass":True,
        "authorized_program_semantic_truth_assumed_externally":True,
        "source_family_summary":summary}


if __name__=="__main__":
    print(json.dumps(audit(),sort_keys=True,indent=2))
