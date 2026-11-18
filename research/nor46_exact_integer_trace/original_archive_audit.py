"""NOR-46 immutable first 16-historical-source exact trace evidence audit.

Checks byte-for-byte first hosted deterministic receipt, fixed old NOR42
source identities, exact-BigInt trace flags and honest/dishonest authority
controls. This never reruns a prover, source generator, or first-grade data.
"""
from __future__ import annotations
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULT=HERE/"results/nor46_first_historical_16.json"
BYTES=19583
SHA256="d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052"
ORIGINAL_NOR45_SHA="c4b1197b87b43c4cb4e32e0c2708c4a070742244d7cc05639e22353af78d5536"


def audit():
    raw=RESULT.read_bytes()
    assert len(raw)==BYTES
    assert sha256(raw).hexdigest()==SHA256
    obj=json.loads(raw)
    assert obj["schema"]=="nor46.historical_full_exact_bigint_trace_16pairs.v1"
    assert obj["original_nor45_first_sha256"]==ORIGINAL_NOR45_SHA
    assert obj["original_nor42_public_source_sha256"]==(
        "f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033")
    assert obj["original_nor42_grade_sha256"]==(
        "bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae")
    assert obj["selected_existing_replicates"]==[100,104,108,112]
    assert obj["historical_pair_count"]==16
    assert obj["new_prospective_nor46_pairs"]==0
    assert obj["reference_external_authority_only"]
    assert obj["all_bigint_arithmetic_checked_without_modular_wrap"]
    assert obj["signed_source_session_query_and_root_checked"]
    assert obj["proof_trace_fully_revealed"]
    assert obj["proof_is_succinct"] is False
    assert obj["proof_is_zero_knowledge"] is False
    assert obj["proof_is_cryptographic_zkvm_or_stark"] is False
    assert obj["original_nor45_sparse_challenges_replayed"]==16
    rows=obj["source_records"]
    assert len(rows)==len({r["source_sha256"] for r in rows})==16
    assert Counter(r["degree"] for r in rows)=={512:8,1024:8}
    assert Counter(r["family"] for r in rows)=={
        "wide_dense_small":8,"wide_regime_blocks":8}
    assert Counter(r["replicate"] for r in rows)=={
        100:4,104:4,108:4,112:4}
    assert obj["reciprocal_R_trace_count"]==sum(
        r["has_reciprocal_R"] for r in rows)==12
    assert all(
        r["full_trace_root_matches_original_nor45"] and
        r["every_native_integer_coefficient_checked"] and
        r["original_nor45_challenge_rejection_reproduced"] and
        r["mutated_trace_rejected"] and
        r["dishonest_signed_output_provider_rejected"] and
        r["dishonest_authorized_clean_program_passes_crypto"] and
        r["dishonest_authorized_program_differs_from_hidden_truth"] and
        r["all_test_private_keys_publicly_known"]
        for r in rows)
    assert obj["mean_full_trace_bytes"]==266511.625
    assert obj["by_degree"]["512"]["mean_full_trace_bytes"]==137099.875
    assert obj["by_degree"]["1024"]["mean_full_trace_bytes"]==395923.375
    return {
        "original_first_receipt_bytes":BYTES,
        "original_first_receipt_sha256":SHA256,
        "old_source_pairs":16,
        "selected_replicates":[100,104,108,112],
        "complete_reciprocal_traces":12,
        "mean_certificate_bytes":obj["mean_full_trace_bytes"],
        "source_model_and_provider_controls_pass":True,
        "source_future_and_full_trace_prover_generated":False,
    }


if __name__=="__main__":
    print(json.dumps(audit(),indent=2,sort_keys=True))
