"""NOR-43 test-only commitment and independent verifier adversarial cases.

No new NOR43 study source identities, private target samples, first-grade
outcome or prospective registered test criteria are generated in these tests.
"""
from __future__ import annotations
import copy
from hashlib import sha256
import random
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from research.nor43_authenticated_witness import signed_witness as producer
from research.nor43_authenticated_witness import verifier_reference as verifier


class WitnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src=sha256(b"synthetic-public-source-ONLY").hexdigest()
        cls.values=tuple((j**3 if j%7 else -(j**2))
                         for j in range(769))
        cls.tree=producer.OutputTree(cls.values)
        cls.key=producer.test_reference_public_key()
        cls.bundle=producer.signed_reference_manifest(
            cls.tree,cls.src,512,signing_key=producer.test_private_key())
        cls.claim=verifier.verify_signed_manifest(
            cls.bundle,cls.key,cls.src,512,len(cls.values),
            producer.PROVIDER_ID)

    def reject(self,bundle=None,public_key=None,source=None,degree=512,
               length=None,provider=None):
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_signed_manifest(
                self.bundle if bundle is None else bundle,
                self.key if public_key is None else public_key,
                self.src if source is None else source,
                degree,len(self.values) if length is None else length,
                producer.PROVIDER_ID if provider is None else provider)

    def test_valid_k16_openings_and_source_context(self):
        self.assertEqual(self.claim["coefficient_count"],769)
        self.assertEqual(self.tree.depth,10)
        self.assertEqual(len(self.tree.root),32)
        self.assertEqual(len(bytes.fromhex(self.bundle["ed25519_signature_hex"])),64)
        for idx in (0,1,2,16,511,512,544,768):
            self.assertTrue(verifier.verify_index_opening(
                self.claim,self.tree.opening(idx)))
        checked=tuple(random.Random(123).sample(range(769),16))
        opens=[self.tree.opening(i) for i in checked]
        self.assertFalse(verifier.verified_sample_rejection(self.values,
            self.claim,opens,challenged_indices=checked))
        altered=list(self.values)
        altered[checked[0]]+=1
        self.assertTrue(verifier.verified_sample_rejection(altered,
            self.claim,opens,challenged_indices=checked))


    def test_all_signed_manifest_context_replays_fail(self):
        self.reject(source=sha256(b"wrong source").hexdigest())
        self.reject(degree=513)
        self.reject(length=1537)
        self.reject(provider="MODEL-SELF-ATTESTED")
        bad=copy.deepcopy(self.bundle)
        bad["manifest"]["protocol"]="NOR43/other-version"
        self.reject(bundle=bad)
        bad=copy.deepcopy(self.bundle)
        bad["manifest"]["source_sha256"]=sha256(b"another source").hexdigest()
        self.reject(bundle=bad)
        bad=copy.deepcopy(self.bundle)
        bad["manifest"]["coefficient_count"]=1537
        self.reject(bundle=bad)
        bad=copy.deepcopy(self.bundle)
        bad["manifest"]["merkle_root_sha256"]="0"*64
        self.reject(bundle=bad)
        bad=copy.deepcopy(self.bundle)
        bad["ed25519_signature_hex"]="00"*64
        self.reject(bundle=bad)
        bad=copy.deepcopy(self.bundle)
        bad["manifest_sha256"]="e"*64
        self.reject(bundle=bad)

    def test_tampered_value_index_sibling_and_noncanonical_decimal(self):
        good=self.tree.opening(544)
        bad=copy.deepcopy(good)
        bad["coefficient"]=str(int(bad["coefficient"])+1)
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)
        bad=copy.deepcopy(good)
        bad["index"]=543
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)
        bad=copy.deepcopy(good)
        bad["siblings"][0]="00"*32
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)
        bad=copy.deepcopy(good)
        bad["siblings"].pop()
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)
        bad=copy.deepcopy(good)
        bad["coefficient"]="+1"
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)
        bad=copy.deepcopy(good)
        bad["coefficient"]="00"
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)
        bad=copy.deepcopy(good)
        bad["index"]=769
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_index_opening(self.claim,bad)

    def test_self_signed_model_root_does_not_authenticate_hidden_truth(self):
        # The public model proposes an internally consistent but WRONG
        # continuation in a hidden target world, then commits to its
        # own false vector using a different model-owned signing key.
        hidden=list(self.values)
        for j in range(544,len(hidden)):
            hidden[j]+=7
        self.assertEqual(hidden[:513],list(self.values[:513]))
        actual=producer.OutputTree(hidden)
        reference=producer.signed_reference_manifest(actual,self.src,512,signing_key=producer.test_private_key())
        authority=verifier.verify_signed_manifest(
            reference,self.key,self.src,512,769,producer.PROVIDER_ID)
        checked=(544,)+tuple(range(15))
        self.assertTrue(verifier.verified_sample_rejection(
            self.values,authority,[actual.opening(i) for i in checked],
            challenged_indices=checked))
        model_key=Ed25519PrivateKey.from_private_bytes(
            sha256(b"PUBLIC-MODEL-SELF-KEY").digest())
        model_bundle=producer.signed_reference_manifest(
            self.tree,self.src,512,signing_key=model_key)
        # With an independently pinned reference key, self-signing fails.
        self.reject(bundle=model_bundle)
        # The insecure arbitrary-key acceptor accepts model's false root.
        from cryptography.hazmat.primitives import serialization
        attacker_pub=model_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        insecure=verifier.verify_signed_manifest(
            model_bundle,attacker_pub,self.src,512,769,producer.PROVIDER_ID)
        self.assertFalse(verifier.verified_sample_rejection(
            self.values,insecure,[self.tree.opening(i) for i in checked],
            challenged_indices=checked))

    def test_authorized_provider_can_sign_wrong_true_world(self):
        """A valid signature authenticates authority, not semantic truth."""
        # Suppose the true world has a hidden suffix, while even the
        # authorized provider mistakenly signs the public clean proposal.
        true_hidden=list(self.values)
        true_hidden[544]+=12345
        clean_reference=producer.signed_reference_manifest(
            self.tree,self.src,512,
            signing_key=producer.test_private_key())
        verified=verifier.verify_signed_manifest(
            clean_reference,self.key,self.src,512,769,producer.PROVIDER_ID)
        requested=(544,)+tuple(range(15))
        openings=[self.tree.opening(i) for i in requested]
        self.assertTrue(self.values[544]!=true_hidden[544])
        self.assertFalse(verifier.verified_sample_rejection(
            self.values,verified,openings,challenged_indices=requested))
        # This is a correct accept under the signed (but FALSE) root.
        # External honest-provider computation is an indispensable premise.
        self.assertEqual(verified["merkle_root_sha256"],self.tree.root.hex())

    def test_proposal_precommitment_before_reference_root(self):
        clean=self.values
        altered=list(clean)
        altered[544]+=1
        self.assertNotEqual(producer.sha_of_proposed_output(clean),
                            producer.sha_of_proposed_output(altered))
        self.assertEqual(producer.sha_of_proposed_output(clean),
                         producer.sha_of_proposed_output(tuple(clean)))

    def test_power_two_and_nonpower_two_trees_large_integers(self):
        for m in (1,2,4,7,8,769,1537):
            rng=random.Random(m)
            values=[0, -1, 1, 2**2048, -(2**4097)] + [
                rng.randrange(-2**40,2**40) for _ in range(max(0,m-5))]
            values=values[:m]
            tree=producer.OutputTree(values)
            bundle=producer.signed_reference_manifest(
                tree,self.src,0,signing_key=producer.test_private_key())
            claim=verifier.verify_signed_manifest(bundle,self.key,
                self.src,0,m,producer.PROVIDER_ID)
            for j in sorted(set((0,m//2,m-1))):
                self.assertTrue(verifier.verify_index_opening(
                    claim,tree.opening(j)))
            self.assertEqual(tree.depth,(m-1).bit_length())
        with self.assertRaises(ValueError):
            producer.OutputTree([])
        with self.assertRaises(ValueError):
            producer.OutputTree([0]).opening(1)


    def test_valid_but_unchallenged_openings_cannot_substitute(self):
        """A model must not choose its own easy indices for a signed root."""
        true_hidden=list(self.values)
        onset=544
        for i in range(onset,len(true_hidden)):
            true_hidden[i]+=5
        tree=producer.OutputTree(true_hidden)
        bundle=producer.signed_reference_manifest(tree,self.src,512,signing_key=producer.test_private_key())
        authorized=verifier.verify_signed_manifest(
            bundle,self.key,self.src,512,769,producer.PROVIDER_ID)
        challenge=(onset,)+tuple(range(15))
        requested=[tree.opening(i) for i in challenge]
        self.assertTrue(verifier.verified_sample_rejection(
            self.values,authorized,requested,challenged_indices=challenge))
        # These substitute proofs are cryptographically VALID under
        # the SAME authorized signed root, but visit easy prefix indices.
        substituted=[tree.opening(i) for i in range(16)]
        for proof in substituted:
            self.assertTrue(verifier.verify_index_opening(authorized,proof))
        with self.assertRaises(verifier.VerificationError):
            verifier.verified_sample_rejection(
                self.values,authorized,substituted,
                challenged_indices=challenge)
        with self.assertRaises(verifier.VerificationError):
            verifier.verified_sample_rejection(
                self.values,authorized,substituted,
                challenged_indices=tuple(range(15))+(0,))

    def test_correct_proofs_are_order_independent_but_challenge_bound(self):
        challenge=tuple(range(16))
        openings=[self.tree.opening(i) for i in reversed(challenge)]
        self.assertFalse(verifier.verified_sample_rejection(
            self.values,self.claim,openings,challenged_indices=challenge))
        with self.assertRaises(TypeError):
            verifier.verified_sample_rejection(
                self.values,self.claim,openings)

    def test_repeated_audit_indices_rejected(self):
        opening=self.tree.opening(0)
        with self.assertRaises(verifier.VerificationError):
            verifier.verified_sample_rejection(
                self.values,self.claim,[opening]*16,
                challenged_indices=tuple(range(16)))


if __name__=="__main__":
    unittest.main()
