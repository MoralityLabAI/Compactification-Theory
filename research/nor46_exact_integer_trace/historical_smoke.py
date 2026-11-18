"""NOR-46 fixed 16 old NOR42 source-pair exact-trace proof smoke.

Historical ONLY: replica groups (100,104,108,112), both D512/D1024,
both dense/block-regime input laws. No new NOR46 source identities or
prospective model outcomes. Original receipt is deterministic and excludes
timing values. All private signing keys used here are PUBLIC test fixtures.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import statistics
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import historical_smoke as hist
from research.nor45_authorized_program import program_authority as authority
from research.nor45_authorized_program import independent_execution_verifier as old
from research.nor46_exact_integer_trace import trace_producer as prover
from research.nor46_exact_integer_trace import trace_verifier as verifier

NOR45_FIRST=ROOT/"research/nor45_authorized_program/results/nor45_first_archived_64_programs.json"
NOR45_FIRST_BYTES=112573
NOR45_FIRST_SHA="c4b1197b87b43c4cb4e32e0c2708c4a070742244d7cc05639e22353af78d5536"
FIXED_REPS=(100,104,108,112)
EPOCH=2


def canonical(data):
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def run():
    original=NOR45_FIRST.read_bytes()
    assert len(original)==NOR45_FIRST_BYTES
    assert sha256(original).hexdigest()==NOR45_FIRST_SHA
    old_receipt=json.loads(original)
    assert old_receipt["schema"]=="nor45.authorized_program_historical_nor42_64_pairs.v1"
    historic={r["original_nor42_source_sha256"]:r
            for r in old_receipt["source_records"]}
    assert len(historic)==64
    fixture,grade=hist.load_original()
    assert len(fixture["cases"])==len(grade["source_records"])==64
    selected=[(row,rec) for row,rec in zip(
        fixture["cases"],grade["source_records"],strict=True)
        if row["pair_replicate"] in FIXED_REPS]
    assert len(selected)==16
    assert Counter(row["degree"] for row,_ in selected)=={512:8,1024:8}
    assert Counter(row["family"] for row,_ in selected)=={
        "wide_dense_small":8,"wide_regime_blocks":8}
    provider_key=trees.test_private_key()
    pinned_provider=trees.test_reference_public_key()
    verifier_key=session.test_verifier_private_key()
    pinned_verifier=session.public_key_bytes(verifier_key)
    authority_key=authority.test_authority_private_key()
    pinned_authority=authority.public_key_of(authority_key)
    records=[]
    for n,(row,graded) in enumerate(selected):
        src=row["full_public_input_sha256"]
        assert graded["public_sha256"]==src
        assert src in historic
        proposal=nor42.nor37.public_proposal(row)
        clean=proposal["pred"]
        query=tuple(row["public_query_input"])
        d=row["degree"]
        clean_truth,hidden_truth,beta,op=nor42.restore_worlds(row,clean)
        assert clean_truth==clean
        assert graded["hidden_beta"]==beta and graded["hidden_transform"]==op
        spec=hist.spec_from_public_proposal(proposal,d,beta,op)
        sess=session.VerifierSession(
            source_sha256=src,observed_degree=d,
            coefficient_count=len(clean),provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=pinned_provider,
            verifier_signing_key=verifier_key,expires_epoch=100,
            nonce=sha256(("NOR46/FIXED-HISTORICAL-TRACE-SESSION/"+src).encode()).digest(),
            test_only=True)
        proposal_sha=session.proposal_digest(clean,sess.context)
        precommit=sess.precommit(proposal_sha,now_epoch=1)
        provider_tree=trees.OutputTree(hidden_truth)
        provider_ref=session.sign_provider_reference(
            provider_tree,precommit,sess.context,pinned_verifier,
            provider_key,now_epoch=EPOCH)
        cert=authority.sign_authorized_program(
            program=spec,query=query,context=sess.context,
            precommit_receipt=precommit,
            pinned_verifier_public_key=pinned_verifier,
            expected_output_merkle_root=provider_tree.root.hex(),
            program_authority_signer=authority_key,now_epoch=EPOCH)
        transcript=prover.transcript(spec,query,d)
        args=dict(certificate=cert,trace=transcript,query=query,
                  session_context=sess.context,precommit_receipt=precommit,
                  provider_reference=provider_ref,
                  pinned_program_authority_key=pinned_authority,
                  expected_program_authority_id=authority.TEST_AUTHORITY_ID,
                  pinned_verifier_public_key=pinned_verifier,
                  pinned_provider_public_key=pinned_provider,now_epoch=EPOCH)
        checked=verifier.verify_authorized_trace(**args)
        assert checked["values"]==hidden_truth
        assert checked["root_sha256"]==provider_tree.root.hex()
        assert checked["trace_sha256"]==prover.certificate_sha256(transcript)
        exact=old.verify_authorized_execution(
            certificate=cert,query=query,
            session_context=sess.context,precommit_receipt=precommit,
            provider_reference=provider_ref,
            pinned_program_authority_key=pinned_authority,
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=pinned_verifier,
            pinned_provider_public_key=pinned_provider,now_epoch=EPOCH)
        assert exact["values"]==checked["values"]
        assert exact["root_sha256"]==checked["root_sha256"]
        # The requested sample comes from a PREVIOUS full NOR45 archived test,
        # rather than selection from any new source, challenge or EVAL result.
        old_challenge=historic[src]["challenge_indices"]
        assert len(set(old_challenge))==16
        sess.authorize_reference(provider_ref,now_epoch=EPOCH)
        sess.challenge(now_epoch=3,test_indices=tuple(old_challenge))
        terminal=sess.finish(
            clean,[provider_tree.opening(i) for i in old_challenge],now_epoch=3)
        sparse=session.verify_terminal_result(
            terminal,sess.context,pinned_verifier)
        support={i for i,(a,b) in enumerate(zip(clean,hidden_truth)) if a!=b}
        assert sparse["reject"]==bool(support.intersection(old_challenge))
        assert sparse["reject"]==historic[src][
            "authenticated_challenge_rejects_clean_proposal"]
        # Valid signed program + fraudulent output provider must fail.
        clean_tree=trees.OutputTree(clean)
        false_provider=session.sign_provider_reference(
            clean_tree,precommit,sess.context,pinned_verifier,
            provider_key,now_epoch=EPOCH)
        try:
            verifier.verify_authorized_trace(**{
                **args,"provider_reference":false_provider})
        except (ValueError,TypeError):
            fraudulent_root_rejected=True
        else:
            fraudulent_root_rejected=False
        assert fraudulent_root_rejected
        # Valid but semantically dishonest authorized CLEAN program is
        # allowed to pass faithful execution and differ from actual hidden
        # truth. This demonstrates the external trust boundary.
        clean_spec=hist.spec_from_public_proposal(proposal,d)
        false_authority_cert=authority.sign_authorized_program(
            program=clean_spec,query=query,context=sess.context,
            precommit_receipt=precommit,
            pinned_verifier_public_key=pinned_verifier,
            expected_output_merkle_root=clean_tree.root.hex(),
            program_authority_signer=authority_key,now_epoch=EPOCH)
        clean_trace=prover.transcript(clean_spec,query,d)
        self_consistent=verifier.verify_authorized_trace(**{
            **args,"certificate":false_authority_cert,
            "provider_reference":false_provider,"trace":clean_trace})
        assert self_consistent["values"]==clean
        assert self_consistent["values"]!=hidden_truth
        # One signed trace coefficient mutation fails the full proof.
        corrupt=copy.deepcopy(transcript)
        corrupt["output_coefficients"][-1]=str(int(
            corrupt["output_coefficients"][-1])+1)
        try:
            verifier.verify_authorized_trace(**{**args,"trace":corrupt})
        except (ValueError,TypeError):
            mutated_trace_rejected=True
        else:
            mutated_trace_rejected=False
        assert mutated_trace_rejected
        records.append({
            "source_sha256":src,
            "replicate":row["pair_replicate"],
            "degree":d,"family":row["family"],
            "future_coefficients":len(clean),
            "authorized_program_base_op":spec["base"],
            "authorized_program_left_op":spec["u"],
            "authorized_program_right_op":spec["v"],
            "authorized_hidden_op":spec["hidden_transform"],
            "unique_transforms_count":len(transcript["unary_transforms"]),
            "has_reciprocal_R":("R" in transcript["unary_transforms"]),
            "full_trace_bytes":prover.trace_json_bytes(transcript),
            "full_trace_sha256":checked["trace_sha256"],
            "signed_program_root_sha256":checked["root_sha256"],
            "full_trace_root_matches_original_nor45":True,
            "every_native_integer_coefficient_checked":True,
            "original_nor45_challenge_rejection_reproduced":True,
            "mutated_trace_rejected":mutated_trace_rejected,
            "dishonest_signed_output_provider_rejected":fraudulent_root_rejected,
            "dishonest_authorized_clean_program_passes_crypto":True,
            "dishonest_authorized_program_differs_from_hidden_truth":True,
            "all_test_private_keys_publicly_known":True,
        })
        print(f"NOR46 archived exact trace {n+1}/16 D{d} "
              f"{row['family']} R={'R' in transcript['unary_transforms']}",
              flush=True)
    assert len(records)==len({r["source_sha256"] for r in records})==16
    assert all(r["full_trace_root_matches_original_nor45"] and
               r["every_native_integer_coefficient_checked"] and
               r["dishonest_authorized_clean_program_passes_crypto"] and
               r["dishonest_authorized_program_differs_from_hidden_truth"] and
               r["mutated_trace_rejected"] for r in records)
    by_degree={str(d):[r for r in records if r["degree"]==d]
               for d in (512,1024)}
    return {
        "schema":"nor46.historical_full_exact_bigint_trace_16pairs.v1",
        "original_nor45_first_sha256":NOR45_FIRST_SHA,
        "original_nor42_public_source_sha256":hist.SOURCE_SHA,
        "original_nor42_grade_sha256":hist.GRADE_SHA,
        "selected_existing_replicates":list(FIXED_REPS),
        "historical_pair_count":16,"new_prospective_nor46_pairs":0,
        "reference_external_authority_only":True,
        "proof_is_succinct":False,"proof_is_zero_knowledge":False,
        "proof_is_cryptographic_zkvm_or_stark":False,
        "proof_trace_fully_revealed":True,
        "all_test_keys_publicly_known":True,
        "signed_source_session_query_and_root_checked":True,
        "all_bigint_arithmetic_checked_without_modular_wrap":True,
        "reciprocal_R_trace_count":sum(r["has_reciprocal_R"] for r in records),
        "original_nor45_sparse_challenges_replayed":16,
        "mean_full_trace_bytes":statistics.mean(
            r["full_trace_bytes"] for r in records),
        "by_degree":{d:{
            "n":len(v),
            "mean_full_trace_bytes":statistics.mean(
                r["full_trace_bytes"] for r in v),
            "reciprocal_R_count":sum(r["has_reciprocal_R"] for r in v)
        } for d,v in by_degree.items()},
        "source_records":records,
        "limitation":"Full exact BigInt trace and root, no succinctness or ZK; "
            "program authority honesty and deployed crypto secrets external. "
            "No CPU timing included in immutable original historical receipt."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=run()
    raw=canonical(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
    print("NOR46 first historical trace bytes SHA",
          len(raw),sha256(raw).hexdigest())
    print("NOR46 original historical summary",
          result["mean_full_trace_bytes"],result["by_degree"],
          flush=True)

if __name__=="__main__":
    main()
