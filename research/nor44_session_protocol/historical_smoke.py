"""NOR-44 historical-only signed-session smoke on original NOR42 pairs.

64 already archived NOR42 sources; zero prospective NOR44 sources or target
worlds generated. Public deterministic test keys/nonces/challenges are
INSECURE REPRODUCIBILITY FIXTURES, not online entropy measurements.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor42_paired_program_ood.prospective import experiment as nor42
from research.nor43_authenticated_witness import historical_smoke as previous
from research.nor43_authenticated_witness import signed_witness as trees
from research.nor44_session_protocol import session_protocol as protocol

FIRST_43_SOURCE_SHA="f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033"
FIRST_43_GRADE_SHA="bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae"
PRIOR_FIRST_SMOKE_SHA="f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145"
TEST_CHALLENGE_STREAM=202611440041
K=16
NOW_EPOCH=11
EXPIRATION_EPOCH=12


def canonical(x):
    return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=True)+"\n").encode()


def seed(*args):
    return int.from_bytes(sha256("|".join(map(str,args)).encode()).digest(),"big")


def nonce_for_archived_source(source):
    """TEST-ONLY deterministic public 256-bit nonce."""
    return sha256(f"NOR44/PUBLIC-HISTORICAL-TEST-NONCE/v1|{source}".encode()).digest()


def exact_first_history():
    fixture,grade=previous.original_first_archive()
    assert fixture["schema"]=="nor42.paired_grammar_ood_public_only_reps100to115.v1"
    assert grade["schema"]=="nor42.paired_program_ood_authenticated_witness.v1"
    assert grade["registered_primary_pass"] is True
    assert fixture["source_units"]==grade["fresh_public_source_pairs"]==64
    assert fixture["excluded_historical_sources"]==388
    return fixture,grade


def historical_session_smoke():
    fixture,grade=exact_first_history()
    provider_key=trees.test_private_key()
    pinned_provider_key=trees.test_reference_public_key()
    verifier_signer=protocol.test_verifier_private_key()
    verifier_public=protocol.public_key_bytes(verifier_signer)
    source_records=[]
    for rank,(row,record) in enumerate(zip(
            fixture["cases"],grade["source_records"],strict=True)):
        src=row["full_public_input_sha256"]
        assert src==record["public_sha256"]
        proposal=nor42.nor37.public_proposal(row)["pred"]
        # Prepare the verifier-owned session and lock first proposal
        # BEFORE revealing any true/reference target coefficients.
        session=protocol.VerifierSession(
            source_sha256=src,
            observed_degree=row["degree"],
            coefficient_count=len(proposal),
            provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=pinned_provider_key,
            verifier_signing_key=verifier_signer,
            expires_epoch=EXPIRATION_EPOCH,
            nonce=nonce_for_archived_source(src),
            test_only=True)
        commit_digest=protocol.proposal_digest(proposal,session.context)
        receipt=session.precommit(commit_digest,now_epoch=NOW_EPOCH)
        verified_receipt=protocol.verify_precommit_receipt(
            receipt,verifier_public,session.context,NOW_EPOCH)
        assert verified_receipt["proposal_sha256"]==commit_digest

        # The hidden truth is available exclusively to the synthetic
        # reference-provider role AFTER proposal commitment was frozen.
        clean,hidden,beta,transform=nor42.restore_worlds(row,proposal)
        assert clean==proposal and beta==record["hidden_beta"]
        assert transform==record["hidden_transform"]
        support={i for i,(a,b) in enumerate(zip(clean,hidden)) if a!=b}
        assert len(support)==record["difference_support_size"]
        tree=trees.OutputTree(hidden)
        provider_bundle=protocol.sign_provider_reference(
            tree,receipt,session.context,verifier_public,
            provider_key,NOW_EPOCH)
        verified_reference=session.authorize_reference(
            provider_bundle,now_epoch=NOW_EPOCH)
        assert verified_reference["proposal_sha256"]==commit_digest
        assert verified_reference["merkle_root_sha256"]==tree.root.hex()

        # PUBLIC deterministic test-only 16 positions (not deployed RNG).
        challenge_indices=tuple(random.Random(
            seed(TEST_CHALLENGE_STREAM,src)).sample(range(len(clean)),K))
        signed_challenge=session.challenge(
            now_epoch=NOW_EPOCH,test_indices=challenge_indices)
        assert signed_challenge["payload"]["requested_indices"]==list(
            challenge_indices)
        opened=[tree.opening(i) for i in challenge_indices]
        final=session.finish(proposal,opened,now_epoch=NOW_EPOCH)
        terminal=protocol.verify_terminal_result(
            final,session.context,verifier_public)
        assert terminal["phase"]=="finalized"
        assert terminal["reject"]==bool(support.intersection(challenge_indices))
        assert session.phase=="finalized"

        # A model cannot replace the nonce-bound proposal after learning
        # the root. Even a valid authorized Merkle opening at an easier
        # coordinate fails if it was NOT in the verifier's challenge.
        second_session=protocol.VerifierSession(
            source_sha256=src,
            observed_degree=row["degree"],
            coefficient_count=len(proposal),
            provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=pinned_provider_key,
            verifier_signing_key=verifier_signer,
            expires_epoch=EXPIRATION_EPOCH,
            nonce=nonce_for_archived_source(src),
            test_only=True)
        pre2=second_session.precommit(commit_digest,now_epoch=NOW_EPOCH)
        root2=protocol.sign_provider_reference(
            tree,pre2,second_session.context,
            verifier_public,provider_key,NOW_EPOCH)
        second_session.authorize_reference(root2,now_epoch=NOW_EPOCH)
        second_session.challenge(now_epoch=NOW_EPOCH,
                                 test_indices=challenge_indices)
        easy=next(i for i in range(len(clean)) if i not in challenge_indices)
        substitutions=[tree.opening(i) for i in range(K)]
        if set(range(K))==set(challenge_indices):
            substitutions[0]=tree.opening(easy)
        assert set(x["index"] for x in substitutions)!=set(challenge_indices)
        try:
            second_session.finish(proposal,substitutions,now_epoch=NOW_EPOCH)
        except (protocol.SessionError,ValueError):
            substitution_rejected=True
        else:
            substitution_rejected=False
        assert substitution_rejected and second_session.phase=="failed"

        # The separately pinned test provider can dishonestly sign the
        # model's WRONG clean vector. Valid cryptographic authorization
        # still does not establish that the committed data are true.
        dishonest=protocol.VerifierSession(
            source_sha256=src,
            observed_degree=row["degree"],
            coefficient_count=len(proposal),
            provider_id=trees.PROVIDER_ID,
            pinned_provider_public_key=pinned_provider_key,
            verifier_signing_key=verifier_signer,
            expires_epoch=EXPIRATION_EPOCH,
            nonce=sha256(b"NOR44/DISHONEST-PROVIDER-TEST/v1"+
                         bytes.fromhex(src)).digest(),
            test_only=True)
        false_digest=protocol.proposal_digest(proposal,dishonest.context)
        false_receipt=dishonest.precommit(false_digest,now_epoch=NOW_EPOCH)
        wrong_tree=trees.OutputTree(proposal)
        false_bundle=protocol.sign_provider_reference(
            wrong_tree,false_receipt,dishonest.context,
            verifier_public,provider_key,NOW_EPOCH)
        dishonest.authorize_reference(false_bundle,now_epoch=NOW_EPOCH)
        dishonest.challenge(now_epoch=NOW_EPOCH,
                            test_indices=challenge_indices)
        dishonest_final=dishonest.finish(
            proposal,[wrong_tree.opening(i) for i in challenge_indices],
            now_epoch=NOW_EPOCH)
        assert dishonest_final["payload"]["reject"] is False
        assert proposal!=hidden

        def size(value):
            return len(json.dumps(
                value,sort_keys=True,separators=(",",":"),
                ensure_ascii=True).encode("ascii"))
        exact=nor42.uniform_exact(len(proposal),len(support))
        assert [exact.numerator,exact.denominator]==record[
            "exact_uniform16_hit_fraction"]
        source_records.append({
            "nor42_public_source_sha256":src,
            "replicate":row["pair_replicate"],
            "degree":row["degree"],
            "family":row["family"],
            "source_output_length":len(proposal),
            "difference_support_size":len(support),
            "session_nonce_hex":session.context["nonce_hex"],
            "test_only_randomness":True,
            "source_session_proposal_sha256":commit_digest,
            "signed_verifier_precommit_sha256":sha256(
                protocol.canonical(receipt)).hexdigest(),
            "signed_provider_reference_sha256":sha256(
                protocol.canonical(provider_bundle)).hexdigest(),
            "signed_challenge_sha256":sha256(
                protocol.canonical(signed_challenge)).hexdigest(),
            "signed_terminal_sha256":sha256(
                protocol.canonical(final)).hexdigest(),
            "challenge_indices_sorted":sorted(challenge_indices),
            "challenge_rejected_wrong_clean_proposal":bool(terminal["reject"]),
            "challenge_intersects_true_difference":bool(
                set(challenge_indices)&support),
            "valid_proof_substitution_rejected":substitution_rejected,
            "dishonest_authorized_provider_signs_false_vector_but_passes_crypto":True,
            "dishonest_authorized_provider_reference_is_true":False,
            "uniform16_exact_detection_fraction":[
                exact.numerator,exact.denominator],
            "signed_verifier_precommit_bytes":size(receipt),
            "signed_provider_reference_bytes":size(provider_bundle),
            "signed_challenge_bytes":size(signed_challenge),
            "signed_terminal_bytes":size(final),
            "sixteen_original_merkle_openings_bytes":size(opened),
            "merkle_depth":tree.depth
        })
        print(f"NOR44 original historical source {rank+1}/64",flush=True)
    assert len(source_records)==64
    assert len({x["nor42_public_source_sha256"] for x in source_records})==64
    assert all(x["valid_proof_substitution_rejected"] and
               x["dishonest_authorized_provider_signs_false_vector_but_passes_crypto"] and
               x["test_only_randomness"] for x in source_records)
    def mean(key,arr):
        return statistics.mean(x[key] for x in arr)
    fields=("signed_verifier_precommit_bytes","signed_provider_reference_bytes",
            "signed_challenge_bytes","signed_terminal_bytes",
            "sixteen_original_merkle_openings_bytes")
    family={}
    for f in ("wide_dense_small","wide_regime_blocks"):
        group=[x for x in source_records if x["family"]==f]
        family[f]={
            "source_pair_count":len(group),
            "fixed_public_test_k16_detection_rate":mean(
                "challenge_rejected_wrong_clean_proposal",group),
            "mean_merkle_depth":mean("merkle_depth",group),
            **{f"mean_{key}":mean(key,group) for key in fields}
        }
    return {
        "schema":"nor44.archived_nor42_session_ordering_smoke.v1",
        "original_nor42_source_sha256":FIRST_43_SOURCE_SHA,
        "original_nor42_grade_sha256":FIRST_43_GRADE_SHA,
        "prior_nor43_original_smoke_sha256":PRIOR_FIRST_SMOKE_SHA,
        "historical_source_pair_count":64,
        "new_nor44_prospective_source_count":0,
        "possible_source_truth_worlds":128,
        "signed_sessions_per_source":3,
        "challenge_queries_per_session":K,
        "first_model_proposals_committed_before_reference":True,
        "reference_roots_signed_under_pinned_test_provider":True,
        "challenge_indices_bound_to_verifier_state":True,
        "all_valid_proof_substitution_attacks_rejected":True,
        "all_dishonest_authorized_provider_cases_crypto_accept_false_truth":True,
        "test_signing_keys_are_public":True,
        "test_nonces_and_challenges_are_public_deterministic":True,
        "families":family,
        "source_records":source_records,
        "scope":"Historical-only synthetic paired worlds, explicit public test "
                "keys and publicly known test randomness; honest provider, "
                "secure key custody and real unpredictability external."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    opts=parser.parse_args()
    result=historical_session_smoke()
    raw=canonical(result)
    if opts.out:
        opts.out.parent.mkdir(parents=True,exist_ok=True)
        opts.out.write_bytes(raw)
    print("NOR44 HISTORICAL-ONLY SESSION FIRST RAW BYTES/SHA",
          len(raw),sha256(raw).hexdigest())
    print("NOR44 HISTORICAL FAMILY MEANS",result["families"])


if __name__=="__main__":
    main()
