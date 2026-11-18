"""NOR-43 permanent first archived-source signed-root evidence byte audit.

Read-only. Regenerates neither source programs nor a different benchmark.
Checks exact original JSON SHA, all 64 archived NOR42 source IDs, the
reference-key boundary and the per-source exact hypergeometric law.
"""
from __future__ import annotations
import json,statistics
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from math import comb,isclose
from pathlib import Path

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/"results/nor43_archived_64_merkle.json"
ORIGINAL_BYTES=110749
ORIGINAL_SHA="f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145"
PAST_SOURCE="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
PAST_GRADE="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"


def audit():
    raw=ORIGINAL.read_bytes()
    assert len(raw)==ORIGINAL_BYTES
    assert sha256(raw).hexdigest()==ORIGINAL_SHA
    grade=json.loads(raw)
    assert grade["schema"]=="nor43.historical_nor42_signed_merkle_reference_smoke.v1"
    assert grade["original_nor42_public_fixture_sha256"]==PAST_SOURCE
    assert grade["original_nor42_first_grade_sha256"]==PAST_GRADE
    assert grade["historical_source_pairs"]==64
    assert grade["new_prospective_nor43_source_units"]==0
    assert grade["challenge_stream"]==202611430041
    assert grade["checked_coefficients_per_pair"]==16
    assert grade["provider_private_key_public_test_only"]
    assert grade["all_proposals_committed_before_true_reference_manifest"]
    assert grade["all_pinned_key_self_signed_attacks_rejected"]
    assert grade["all_witnessfree_hidden_selfconsistency_controls_accept"]
    rows=grade["source_records"]
    ids=[r["original_nor42_public_sha256"] for r in rows]
    assert len(ids)==len(set(ids))==64
    assert Counter(r["original_replicate"] for r in rows)=={
        rep:4 for rep in range(100,116)}
    assert Counter(r["degree"] for r in rows)=={512:32,1024:32}
    assert Counter(r["family"] for r in rows)=={
        "wide_dense_small":32,"wide_regime_blocks":32}
    assert all(r["test_provider_public_key_sha256"]==
               grade["synthetic_test_reference_key_public_sha256"]
               for r in rows)
    for r in rows:
        assert r["hidden_world_equal_public_prefix"]
        assert r["self_signed_model_root_rejected_by_pinned_key"]
        assert r["self_signed_model_root_passes_insecure_arbitrary_key_verifier"]
        assert r["witness_free_clean_self_consistency_accepts_hidden_world"]
        assert len(r["historical_k16_checked_indices"])==16
        assert len(set(r["historical_k16_checked_indices"]))==16
        m=r["future_length"]
        s=r["hidden_difference_support_size"]
        assert m==3*r["degree"]//2+1
        assert 1<=s<=m and r["first_hidden_difference_coefficient"]==r["degree"]+32
        assert r["merkle_depth"]==(m-1).bit_length()
        assert r["sixteen_openings_payload_bytes"]>16*32*r["merkle_depth"]
        assert r["signed_manifest_payload_bytes"]==569
        assert r["bytes_per_opening_mean"]==r["sixteen_openings_payload_bytes"]/16
        exact=Fraction(1)-Fraction(comb(m-s,16),comb(m,16)) if m-s>=16 else Fraction(1)
        assert r["true_hidden_uniform16_exact_hit_fraction"]==[
            exact.numerator,exact.denominator]
        assert isclose(float(exact),
                       r["true_hidden_uniform16_exact_hit_probability"],
                       abs_tol=1e-15)
        assert all(0<=i<m for i in r["historical_k16_checked_indices"])
        assert len(r["true_hidden_root_sha256"])==64
        assert len(r["proposed_clean_output_sha256"])==64
    hits=sum(r["challenge_draw_hits_hidden_difference_support"] for r in rows)
    assert hits==57
    for family in ("wide_dense_small","wide_regime_blocks"):
        sub=[r for r in rows if r["family"]==family]
        prev=grade["family_means"][family]
        assert prev["source_pairs"]==32
        assert isclose(prev["uniform16_exact_support_hit_mean"],
                       statistics.mean(r["true_hidden_uniform16_exact_hit_probability"]
                                       for r in sub),abs_tol=1e-15)
        assert prev["fixed_seed_k16_hidden_hit_rate"]==statistics.mean(
            r["challenge_draw_hits_hidden_difference_support"] for r in sub)
        assert prev["mean_sixteen_proof_bytes"]==statistics.mean(
            r["sixteen_openings_payload_bytes"] for r in sub)
        assert prev["mean_signed_manifest_bytes"]==569
    return {
        "schema":"nor43.original_signed_merkle_immutable_archive_audit.v1",
        "original_raw_bytes":ORIGINAL_BYTES,
        "original_raw_sha256":ORIGINAL_SHA,
        "historical_source_pairs":64,
        "new_prospective_source_ids":0,
        "reference_signature_boundary_cases":64,
        "signed_hidden_world_checked_hits":57,
        "family_means":grade["family_means"]
    }


if __name__=="__main__":
    print(json.dumps(audit(),sort_keys=True,indent=2))
