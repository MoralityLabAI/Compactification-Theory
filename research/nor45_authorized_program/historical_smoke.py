"""NOR45 historical-only signed program/output root integration on 64 NOR42 pairs.

NO new NOR45 source identifiers, program-family test corpus or prospective
scientific outcomes. This evaluates exactly the already archived NOR42
true hidden worlds, using PUBLIC test signing keys and deterministic public
challenge fixtures. Honest program authority is assumed in the primary arm;
the dishonest authorized authority is an explicit separate negative control.

No wall clock data are put in the immutable first receipt.
"""
from __future__ import annotations
import argparse,json,random,sys
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
from fractions import Fraction
from statistics import mean
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as session
from research.nor45_authorized_program import program_authority as authority
from research.nor45_authorized_program import independent_execution_verifier as verifier

FIRST_ARCHIVE=ROOT/"research/nor42_paired_program_ood/prospective/results/nor42_original_first_hosted.zip"
SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
GRADE_SHA="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
ZIP_SHA="e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d"
ZIP_BYTES=2799867
STREAM=202611450041
S_NONCE=202611450042
EPOCH=1


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode("ascii")


def load_original():
    archive=FIRST_ARCHIVE.read_bytes()
    assert len(archive)==ZIP_BYTES and sha256(archive).hexdigest()==ZIP_SHA
    with ZipFile(FIRST_ARCHIVE) as z:
        assert z.testzip() is None
        src=z.read("nor42_fixture.json")
        grade=z.read("nor42_full.json")
    assert len(src)==9059929 and sha256(src).hexdigest()==SOURCE_SHA
    assert len(grade)==201956 and sha256(grade).hexdigest()==GRADE_SHA
    fixture=json.loads(src);original=json.loads(grade)
    assert fixture["registered_first_commit"]==original["registration_first_commit"]
    assert fixture["source_units"]==original["fresh_public_source_pairs"]==64
    assert original["registered_primary_pass"] is True
    return fixture,original


def _digest(*parts):
    return sha256("|".join(map(str,parts)).encode("ascii")).digest()


def spec_from_public_proposal(proposal,hidden_beta=0,hidden_op=None):
    h=proposal["header"]
    cls=proposal["class"]
    native=tuple(nor42.corpus.OPS)
    assert cls is not None and 0<=cls<16
    result={
        "base":h.base,"delay":h.delay,"spacing":h.spacing,
        "amplitude":h.amplitude,
        "u":native[cls//4],"v":native[cls%4],
        "hidden_beta":hidden_beta,"hidden_transform":hidden_op
    }
    authority.validate_program_spec(result,512 if h.delay<=112 else 1024)
    return result


def bytes_len(obj):
    return len(json.dumps(obj,sort_keys=True,separators=(",",":"),
                          ensure_ascii=True).encode("ascii"))


def smoke():
    fixture,grade=load_original()
    provider_signer=trees.test_private_key()
    provider_pub=trees.test_reference_public_key()
    verifier_signer=session.test_verifier_private_key()
    verifier_pub=session.public_key_bytes(verifier_signer)
    trusted_authority=authority.test_authority_private_key()
    authority_pub=authority.public_key_of(trusted_authority)
    model_signer=Ed25519PrivateKey.from_private_bytes(
        sha256(b"NOR45/UNTRUSTED-PUBLIC-MODEL-SIGNER/v1").digest())
    assert authority_pub!=authority.public_key_of(model_signer)
    records=[]
    for n,(row,original) in enumerate(zip(
            fixture["cases"],grade["source_records"],strict=True)):
        src=row["full_public_input_sha256"]
        assert src==original["public_sha256"]
        proposal=nor42.nor37.public_proposal(row)
        clean=proposal["pred"]
        query=tuple(row["public_query_input"])
        assert proposal["sha"]==original["clean_model_prediction_sha256"]
        spec_clean=spec_from_public_proposal(proposal)
        sess=session.VerifierSession(
            source_sha256=src,observed_degree=row["degree"],
            coefficient_count=len(clean),
            provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=provider_pub,
            verifier_signing_key=verifier_signer,
            expires_epoch=100,
            nonce=_digest(S_NONCE,src),test_only=True)
        # Freeze first proposal receipt before opening ANY hidden world.
        proposed_sha=session.proposal_digest(clean,sess.context)
        receipt=sess.precommit(proposed_sha,now_epoch=EPOCH)
        assert receipt["payload"]["proposal_sha256"]==proposed_sha
        clean_true,hidden_true,beta,hidden_op=nor42.restore_worlds(row,clean)
        assert clean_true==clean
        assert beta==original["hidden_beta"] and hidden_op==original["hidden_transform"]
        assert nor42.nor24.sha_target(hidden_true)==original[
            "private_hidden_truth_sha256_after_freeze"]
        spec_hidden={**spec_clean,
                     "hidden_beta":beta,"hidden_transform":hidden_op}
        authority.validate_program_spec(spec_hidden,row["degree"])
        hidden_tree=trees.OutputTree(hidden_true)
        clean_tree=trees.OutputTree(clean_true)
        provider_ref=session.sign_provider_reference(
            hidden_tree,receipt,sess.context,verifier_pub,
            provider_signer,now_epoch=2)
        certificate=authority.sign_authorized_program(
            program=spec_hidden,query=query,context=sess.context,
            precommit_receipt=receipt,pinned_verifier_public_key=verifier_pub,
            expected_output_merkle_root=hidden_tree.root.hex(),
            program_authority_signer=trusted_authority,now_epoch=2)
        verified=verifier.verify_authorized_execution(
            certificate=certificate,query=query,
            session_context=sess.context,precommit_receipt=receipt,
            provider_reference=provider_ref,
            pinned_program_authority_key=authority_pub,
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=verifier_pub,
            pinned_provider_public_key=provider_pub,now_epoch=2)
        assert verified["values"]==hidden_true
        assert verified["root_sha256"]==hidden_tree.root.hex()
        sess.authorize_reference(provider_ref,now_epoch=2)
        checked=tuple(random.Random(
            int.from_bytes(_digest(STREAM,src),"big")).sample(
                range(len(clean)),16))
        sess.challenge(now_epoch=3,test_indices=checked)
        result=sess.finish(
            clean,[hidden_tree.opening(i) for i in checked],now_epoch=3)
        terminal=session.verify_terminal_result(
            result,sess.context,verifier_pub)
        support=frozenset(i for i,(a,b) in enumerate(zip(
            clean,hidden_true)) if a!=b)
        assert len(support)==original["difference_support_size"]
        assert terminal["reject"]==bool(support.intersection(checked))
        assert terminal["phase"]=="finalized"

        # An authorized but DISHONEST OUTPUT provider signs clean output
        # despite a separately certified hidden executable program.
        false_root_reference=session.sign_provider_reference(
            clean_tree,receipt,sess.context,verifier_pub,
            provider_signer,now_epoch=2)
        assert session.verify_provider_reference(
            false_root_reference,sess.context,receipt,
            provider_pub,verifier_pub,now_epoch=2)["merkle_root_sha256"]==(
            clean_tree.root.hex())
        try:
            verifier.verify_authorized_execution(
                certificate=certificate,query=query,
                session_context=sess.context,precommit_receipt=receipt,
                provider_reference=false_root_reference,
                pinned_program_authority_key=authority_pub,
                expected_program_authority_id=authority.TEST_AUTHORITY_ID,
                pinned_verifier_public_key=verifier_pub,
                pinned_provider_public_key=provider_pub,now_epoch=2)
        except verifier.ProgramAttestationError:
            output_fraud_rejected=True
        else:output_fraud_rejected=False
        assert output_fraud_rejected

        # A model-owned self-signed CLEAN program root does not have the
        # separately pinned authority's signature.
        false_model_certificate=authority.sign_authorized_program(
            program=spec_clean,query=query,context=sess.context,
            precommit_receipt=receipt,pinned_verifier_public_key=verifier_pub,
            expected_output_merkle_root=clean_tree.root.hex(),
            program_authority_signer=model_signer,now_epoch=2)
        try:
            verifier.verify_authorized_execution(
                certificate=false_model_certificate,query=query,
                session_context=sess.context,precommit_receipt=receipt,
                provider_reference=false_root_reference,
                pinned_program_authority_key=authority_pub,
                expected_program_authority_id=authority.TEST_AUTHORITY_ID,
                pinned_verifier_public_key=verifier_pub,
                pinned_provider_public_key=provider_pub,now_epoch=2)
        except verifier.ProgramAttestationError:
            model_selfroot_rejected=True
        else:model_selfroot_rejected=False
        assert model_selfroot_rejected

        # The AUTHORIZED program authority can be dishonest too. If it
        # chooses clean program, a faithful clean output root passes all
        # signatures and independent execution, while truth was hidden.
        dishonest_authority_certificate=authority.sign_authorized_program(
            program=spec_clean,query=query,context=sess.context,
            precommit_receipt=receipt,pinned_verifier_public_key=verifier_pub,
            expected_output_merkle_root=clean_tree.root.hex(),
            program_authority_signer=trusted_authority,now_epoch=2)
        false_but_auth=verifier.verify_authorized_execution(
            certificate=dishonest_authority_certificate,query=query,
            session_context=sess.context,precommit_receipt=receipt,
            provider_reference=false_root_reference,
            pinned_program_authority_key=authority_pub,
            expected_program_authority_id=authority.TEST_AUTHORITY_ID,
            pinned_verifier_public_key=verifier_pub,
            pinned_provider_public_key=provider_pub,now_epoch=2)
        dishonest_authority_semantic_failure=(
            false_but_auth["values"]!=hidden_true and
            false_but_auth["root_sha256"]==clean_tree.root.hex())
        assert dishonest_authority_semantic_failure

        exact=nor42.uniform_exact(len(clean),len(support))
        assert [exact.numerator,exact.denominator]==original[
            "exact_uniform16_hit_fraction"]
        records.append({
            "original_nor42_source_sha256":src,
            "replicate":row["pair_replicate"],
            "degree":row["degree"],"family":row["family"],
            "future_coefficients":len(clean),
            "original_hidden_difference_support_size":len(support),
            "precommit_source_and_session_bound":True,
            "independent_program_reexecution_matches_hidden_true_world":True,
            "independently_rebuilt_hidden_root_matches_publisher":True,
            "exact_hidden_uniform16_probability_fraction":[
                exact.numerator,exact.denominator],
            "challenge_indices":sorted(checked),
            "authenticated_challenge_rejects_clean_proposal":terminal["reject"],
            "unauthorized_model_program_rejected":model_selfroot_rejected,
            "dishonest_authorized_output_provider_rejected":output_fraud_rejected,
            "dishonest_authorized_program_semantic_failure":(
                dishonest_authority_semantic_failure),
            "signed_program_certificate_bytes":bytes_len(certificate),
            "verifier_signed_first_proposal_receipt_bytes":bytes_len(receipt),
            "provider_signed_reference_bytes":bytes_len(provider_ref),
            "sixteen_merkle_openings_bytes":bytes_len(
                [hidden_tree.opening(i) for i in checked]),
            "signed_terminal_receipt_bytes":bytes_len(result),
            "all_test_private_keys_publicly_known":True,
            "test_challenge_is_publicly_reproducible":True,
            "session_bound_to_original_source":sess.context["source_sha256"]==src,
            "independent_program_authority_key_sha256":sha256(
                authority_pub).hexdigest(),
            "original_public_clean_prediction_sha256":proposal["sha"],
        })
        print(f"NOR45 archived pair {n+1}/64",flush=True)
    assert len(records)==len({a["original_nor42_source_sha256"] for a in records})==64
    families={name:[r for r in records if r["family"]==name]
              for name in ("wide_dense_small","wide_regime_blocks")}
    summary={
        name:{
            "n":len(rows),
            "sampled_k16_detection":sum(
                r["authenticated_challenge_rejects_clean_proposal"] for r in rows),
            "exact_k16_detection_mean":mean(
                float(Fraction(*r["exact_hidden_uniform16_probability_fraction"]))
                for r in rows),
            "mean_program_cert_bytes":mean(
                r["signed_program_certificate_bytes"] for r in rows),
            "mean_proof_bytes":mean(
                r["sixteen_merkle_openings_bytes"] for r in rows),
        } for name,rows in families.items()}
    assert all(result["n"]==32 for result in summary.values())
    assert all(r["dishonest_authorized_output_provider_rejected"] and
               r["unauthorized_model_program_rejected"] and
               r["dishonest_authorized_program_semantic_failure"] and
               r["independently_rebuilt_hidden_root_matches_publisher"] and
               r["independent_program_reexecution_matches_hidden_true_world"]
               for r in records)
    return {
        "schema":"nor45.authorized_program_historical_nor42_64_pairs.v1",
        "original_nor42_source_sha256":SOURCE_SHA,
        "original_nor42_grade_sha256":GRADE_SHA,
        "historical_original_source_pairs":64,
        "new_nor45_prospective_source_ids":0,
        "program_authority_private_key_public_test_fixture":True,
        "verifier_test_challenge_predictable":True,
        "all_authorized_program_executions_match_signed_hidden_reference":True,
        "all_dishonest_signed_output_roots_rejected_by_program_recomputation":True,
        "all_model_self_signed_programs_rejected":True,
        "all_dishonest_authorized_programs_pass_crypto_but_miss_true_world":True,
        "family_summaries":summary,"source_records":records,
        "scope":"Historical synthetic signed-program and full CPU exact recomputation. "
                "Program authority correctness and external true world remain "
                "trust premises. No succinct or cryptographic proof of evaluation."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=smoke()
    raw=canonical(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(raw)
    print("NOR45 ORIGINAL HISTORICAL SMOKE BYTES SHA256",
          len(raw),sha256(raw).hexdigest())
    print("NOR45 FAMILY SUMMARY",result["family_summaries"])


if __name__=="__main__":
    main()
