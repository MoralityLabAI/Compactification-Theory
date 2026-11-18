"""NOR-48 fixed original 16 NOR42 source/program/session cases.

Source IDs and original trace/root/performance receipts are committed and
SHA-locked in NOR46/NOR47. All signing keys are PUBLIC test-only fixtures.
No new source identities, program hypotheses or model grades are generated.
"""
from __future__ import annotations
import json
from collections import Counter
from hashlib import sha256

from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import historical_smoke as hist
from research.nor45_authorized_program import program_authority as authority
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_producer as proof
from research.nor46_exact_integer_trace import original_archive_audit as lock46
from research.nor47_gmp_exact_convolution import original_archive_audit as lock47

EPOCH=2


def historical_signed_16():
    checked46=lock46.audit()
    checked47=lock47.audit()
    assert checked46["old_source_pairs"]==checked47["fixed_historical_source_pairs"]==16
    first=json.loads(lock46.RESULT.read_bytes())
    authorized={r["source_sha256"]:r for r in first["source_records"]}
    fixture,graded=hist.load_original()
    assert len(authorized)==16
    selected=[(row,record) for row,record in zip(
        fixture["cases"],graded["source_records"],strict=True)
        if row["full_public_input_sha256"] in authorized]
    assert len(selected)==16
    assert Counter(row["pair_replicate"] for row,_ in selected)=={
        100:4,104:4,108:4,112:4}
    provider_key=trees.test_private_key()
    provider_public=trees.test_reference_public_key()
    verifier_key=session.test_verifier_private_key()
    verifier_public=session.public_key_bytes(verifier_key)
    program_key=authority.test_authority_private_key()
    program_public=authority.public_key_of(program_key)
    cases=[]
    for row,receipt in selected:
        sha=row["full_public_input_sha256"]
        assert receipt["public_sha256"]==sha
        proposal=nor42.nor37.public_proposal(row)
        clean=proposal["pred"]
        _,true_output,beta,op=nor42.restore_worlds(row,clean)
        native_program=hist.spec_from_public_proposal(
            proposal,row["degree"],beta,op)
        query=tuple(row["public_query_input"])
        sess=session.VerifierSession(
            source_sha256=sha,observed_degree=row["degree"],
            coefficient_count=len(clean),provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=provider_public,
            verifier_signing_key=verifier_key,
            expires_epoch=100,
            nonce=sha256(("NOR48/fixed-source-authorization/"+sha).encode()).digest(),
            test_only=True)
        precommit=sess.precommit(
            session.proposal_digest(clean,sess.context),now_epoch=1)
        hidden_root=trees.OutputTree(true_output)
        signed_provider=session.sign_provider_reference(
            hidden_root,precommit,sess.context,verifier_public,
            provider_key,now_epoch=EPOCH)
        certificate=authority.sign_authorized_program(
            program=native_program,query=query,context=sess.context,
            precommit_receipt=precommit,
            pinned_verifier_public_key=verifier_public,
            expected_output_merkle_root=hidden_root.root.hex(),
            program_authority_signer=program_key,
            now_epoch=EPOCH)
        full_trace=proof.transcript(native_program,query,row["degree"])
        assert full_trace["output_coefficients"]==[str(x) for x in true_output]
        assert hidden_root.root.hex()==authorized[sha]["signed_program_root_sha256"]
        required={native_program[k] for k in ("base","u","v")}
        if native_program["hidden_beta"]:
            required.add(native_program["hidden_transform"])
        contains_R=("R" in required)
        assert contains_R==authorized[sha]["has_reciprocal_R"]
        options={
            "certificate":certificate,"trace_witness":full_trace,
            "query":query,"session_context":sess.context,
            "precommit_receipt":precommit,
            "provider_reference":signed_provider,
            "pinned_program_authority_key":program_public,
            "expected_program_authority_id":authority.TEST_AUTHORITY_ID,
            "pinned_verifier_public_key":verifier_public,
            "pinned_provider_public_key":provider_public,
            "now_epoch":EPOCH
        }
        cases.append({
            "source_sha256":sha,"replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "requires_R":contains_R,
            "true_output":true_output,
            "signed_root":hidden_root.root.hex(),
            "signed_program":native_program,
            "query":query,
            "trace_bytes":proof.trace_json_bytes(full_trace),
            "opts":options
        })
    assert Counter(x["requires_R"] for x in cases)=={True:12,False:4}
    return cases
