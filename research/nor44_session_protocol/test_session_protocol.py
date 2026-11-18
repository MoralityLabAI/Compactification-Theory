"""NOR-44 verifier-session protocol ordering, signatures and attack controls.

Uses artificial public source strings and a public, deterministic TEST key.
No new prospective NOR44 source identities or experimental truth worlds.
"""
from __future__ import annotations

import copy
import hashlib
import secrets
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from research.nor43_authenticated_witness import signed_witness as root_producer
from research.nor44_session_protocol import session_protocol as p


class NOR44SyntheticSessionTest(unittest.TestCase):
    def setUp(self):
        self.src=hashlib.sha256(b"NOR44/SYNTHETIC-PUBLIC-CONTEXT").hexdigest()
        self.clean=tuple((n*7 if n%3 else -(n**2)) for n in range(769))
        hidden=list(self.clean)
        for n in range(544,769):hidden[n]+=7
        self.hidden=tuple(hidden)
        self.true_tree=root_producer.OutputTree(self.hidden)
        self.clean_tree=root_producer.OutputTree(self.clean)
        self.provider=root_producer.test_private_key()
        self.pinned=root_producer.test_reference_public_key()
        self.verifier=p.test_verifier_private_key()
        self.challenge=(544,)+tuple(range(15))

    def session(self,*,nonce=None,test_only=True,expiry=10,
                source=None,degree=512,length=769,provider_id=None):
        return p.VerifierSession(
            source_sha256=self.src if source is None else source,
            observed_degree=degree,
            coefficient_count=length,
            provider_id=root_producer.PROVIDER_ID if provider_id is None else provider_id,
            pinned_provider_public_key=self.pinned,
            verifier_signing_key=self.verifier,
            expires_epoch=expiry,
            nonce=bytes(range(32)) if nonce is None and test_only else nonce,
            test_only=test_only)

    def prepare(self,s,tree=None):
        clean_hash=p.proposal_digest(self.clean,s.context)
        receipt=s.precommit(clean_hash,now_epoch=7)
        self.assertEqual(s.phase,"proposed")
        key=p.public_key_bytes(self.verifier)
        self.assertEqual(p.verify_precommit_receipt(
            receipt,key,s.context,7)["proposal_sha256"],clean_hash)
        provider_root=p.sign_provider_reference(
            self.true_tree if tree is None else tree,
            receipt,s.context,key,self.provider,7)
        s.authorize_reference(provider_root,now_epoch=7)
        self.assertEqual(s.phase,"reference_authenticated")
        return receipt,provider_root

    def test_honest_hidden_world_rejected_and_signed_result(self):
        s=self.session()
        receipt,root=self.prepare(s)
        self.assertNotIn("merkle_root_sha256",receipt["payload"])
        challenge=s.challenge(now_epoch=7,test_indices=self.challenge)
        self.assertEqual(challenge["payload"]["requested_indices"],
                         list(self.challenge))
        self.assertEqual(s.phase,"challenged")
        openings=[self.true_tree.opening(i) for i in self.challenge]
        final=s.finish(self.clean,openings,now_epoch=7)
        self.assertEqual(s.phase,"finalized")
        verified=p.verify_terminal_result(
            final,s.context,p.public_key_bytes(self.verifier))
        self.assertIs(verified["reject"],True)
        self.assertEqual(verified["phase"],"finalized")
        self.assertEqual(verified["requested_indices"],list(self.challenge))
        with self.assertRaises(p.SessionError):
            s.finish(self.clean,openings,now_epoch=7)

    def test_honest_clean_world_accepted(self):
        s=self.session()
        self.prepare(s,tree=self.clean_tree)
        s.challenge(now_epoch=7,test_indices=self.challenge)
        final=s.finish(self.clean,[self.clean_tree.opening(i)
                                   for i in self.challenge],now_epoch=7)
        self.assertIs(final["payload"]["reject"],False)

    def test_model_cannot_equivocate_after_committing(self):
        s=self.session()
        self.prepare(s)
        with self.assertRaises(p.SessionError):
            s.precommit(p.proposal_digest(self.hidden,s.context),now_epoch=7)
        self.assertEqual(s.phase,"failed")
        with self.assertRaises(p.SessionError):
            s.challenge(now_epoch=7,test_indices=self.challenge)
        # Separate valid session: a changed post-root reveal fails and cannot retry.
        fresh=self.session(nonce=b"r"*32)
        self.prepare(fresh)
        fresh.challenge(now_epoch=7,test_indices=self.challenge)
        with self.assertRaises(p.SessionError):
            fresh.finish(self.hidden,[self.true_tree.opening(i)
                                    for i in self.challenge],now_epoch=7)
        self.assertEqual(fresh.phase,"failed")
        with self.assertRaises(p.SessionError):
            fresh.finish(self.clean,[self.true_tree.opening(i)
                                    for i in self.challenge],now_epoch=7)

    def test_unrequested_individually_valid_proofs_fail(self):
        s=self.session()
        self.prepare(s)
        s.challenge(now_epoch=7,test_indices=self.challenge)
        easy=[self.true_tree.opening(i) for i in range(16)]
        for proof in easy:
            self.assertTrue(root_producer.OutputTree(self.hidden).root==
                            self.true_tree.root)
        with self.assertRaises(Exception):
            s.finish(self.clean,easy,now_epoch=7)
        self.assertEqual(s.phase,"failed")

    def test_challenge_injection_works_only_in_explicit_test_mode(self):
        with self.assertRaises(p.SessionError):
            self.session(nonce=bytes(range(32)),test_only=False)
        s=self.session(test_only=False)
        self.assertFalse(s.context["test_only"])
        self.assertEqual(len(bytes.fromhex(s.context["nonce_hex"])),32)
        self.prepare(s)
        with self.assertRaises(p.SessionError):
            s.challenge(now_epoch=7,test_indices=self.challenge)
        self.assertEqual(s.phase,"failed")

    def test_real_csprng_challenge_path_and_proposal_binding(self):
        s=self.session(test_only=False)
        self.prepare(s)
        signed_challenge=s.challenge(now_epoch=7)
        chosen=s.requested_indices
        self.assertEqual(len(chosen),16)
        self.assertEqual(len(set(chosen)),16)
        self.assertEqual(signed_challenge["payload"]["requested_indices"],
                         list(chosen))
        final=s.finish(self.clean,[self.true_tree.opening(i) for i in chosen],
                       now_epoch=7)
        self.assertEqual(final["payload"]["reject"],
                         bool(set(chosen).intersection(range(544,769))))

    def test_second_session_and_reference_replays_fail(self):
        a=self.session()
        pre,reference=self.prepare(a)
        b=self.session(nonce=b"b"*32)
        b.precommit(p.proposal_digest(self.clean,b.context),now_epoch=7)
        with self.assertRaises(p.SessionError):
            b.authorize_reference(reference,now_epoch=7)
        self.assertEqual(b.phase,"failed")
        bad=copy.deepcopy(reference)
        bad["payload"]["context"]["source_sha256"]="0"*64
        c=self.session(nonce=b"c"*32)
        self.prepare(c)
        with self.assertRaises(p.SessionError):
            c.authorize_reference(bad,now_epoch=7)

    def test_source_degree_length_provider_and_nonce_are_bound(self):
        base=self.session()
        receipt,reference=self.prepare(base)
        for changes in (
            {"source":hashlib.sha256(b"other").hexdigest()},
            {"degree":511},{"length":1537},
            {"provider_id":"ATTACKER-CONTROL"},
            {"nonce":b"z"*32}):
            s=self.session(**changes)
            # Wrong length requires matching proposal hash; receipt still
            # cryptographically mismatches even if its output reveal differs.
            s.precommit("0"*64,now_epoch=7)
            with self.assertRaises(p.SessionError):
                s.authorize_reference(reference,now_epoch=7)
        # Signature integrity still holds for genuine earlier receipt.
        self.assertIsNotNone(p.verify_precommit_receipt(
            receipt,p.public_key_bytes(self.verifier),base.context,7))

    def test_expired_epoch_rejected(self):
        s=self.session(expiry=7)
        pre,_=self.prepare(s)
        with self.assertRaises(p.SessionError):
            s.challenge(now_epoch=8)
        with self.assertRaises(p.SessionError):
            p.verify_precommit_receipt(
                pre,p.public_key_bytes(self.verifier),s.context,8)

    def test_invalid_provider_signature_and_dishonest_authorized_root(self):
        model_key=Ed25519PrivateKey.from_private_bytes(
            hashlib.sha256(b"NOR44/MODEL-SELF-SIGNER").digest())
        s=self.session()
        receipt=s.precommit(p.proposal_digest(self.clean,s.context),now_epoch=7)
        with self.assertRaises(p.SessionError):
            p.sign_provider_reference(self.clean_tree,receipt,s.context,
                p.public_key_bytes(self.verifier),model_key,7)
        # The genuine provider can sign the WRONG vector. Its signature
        # checks, illustrating that semantic provider honesty is external.
        signed_wrong=p.sign_provider_reference(
            self.clean_tree,receipt,s.context,
            p.public_key_bytes(self.verifier),self.provider,7)
        s.authorize_reference(signed_wrong,now_epoch=7)
        s.challenge(now_epoch=7,test_indices=self.challenge)
        result=s.finish(self.clean,[self.clean_tree.opening(i)
                                   for i in self.challenge],now_epoch=7)
        self.assertFalse(result["payload"]["reject"])
        self.assertNotEqual(self.clean,self.hidden)

    def test_reference_stage_before_proposal_is_forbidden(self):
        s=self.session()
        arbitrary={"payload":{"protocol":p.REFERENCE_PROTOCOL},
                   "signature_hex":"00"*64}
        with self.assertRaises(p.SessionError):
            s.authorize_reference(arbitrary,now_epoch=7)
        with self.assertRaises(p.SessionError):
            s.challenge(now_epoch=7)
        with self.assertRaises(p.SessionError):
            s.finish(self.clean,[],now_epoch=7)

    def test_reference_tamper_and_receipt_tamper(self):
        s=self.session()
        proposal=p.proposal_digest(self.clean,s.context)
        receipt=s.precommit(proposal,now_epoch=7)
        forged=copy.deepcopy(receipt)
        forged["payload"]["proposal_sha256"]=p.proposal_digest(
            self.hidden,s.context)
        with self.assertRaises(p.SessionError):
            p.verify_precommit_receipt(
                forged,p.public_key_bytes(self.verifier),s.context,7)
        ref=p.sign_provider_reference(self.true_tree,receipt,s.context,
            p.public_key_bytes(self.verifier),self.provider,7)
        forged=copy.deepcopy(ref)
        forged["payload"]["merkle_root_sha256"]="0"*64
        with self.assertRaises(p.SessionError):
            s.authorize_reference(forged,now_epoch=7)
        self.assertEqual(s.phase,"failed")

    def test_signed_receipt_test_only_flag_and_noncanonical_context(self):
        s=self.session()
        pre=s.precommit(p.proposal_digest(self.clean,s.context),now_epoch=7)
        self.assertIs(pre["payload"]["context"]["test_only"],True)
        for damaged in (
            {"source_sha256":self.src.upper()},
            {"observed_degree":True},
            {"coefficient_count":False},
            {"test_only":"false"}):
            c=dict(s.context,counter="invalid")
            c.pop("counter")
            c.update(damaged)
            with self.assertRaises(p.SessionError):
                p._validate_context(c)
        with self.assertRaises(p.SessionError):
            p.proposal_digest(tuple(self.clean[:-1]),s.context)


if __name__=="__main__":
    unittest.main()
