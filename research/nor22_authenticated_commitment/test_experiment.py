"""NOR-22 exact fingerprint and frozen-world policy safety checks."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from research.nor22_authenticated_commitment import experiment as n
from research.rrc_compactification import compound25 as corpus


class NOR22AuthenticatedCommitmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case,cls.header,cls.planted,cls.attempt,cls.digest,cls.mask=(
            n.generate_new_source(0,512,"wide_sparse_small"))
        cls.real_future=corpus.execute(cls.case.query,cls.planted)
        cls.public=n.select_public_candidate(
            cls.case,cls.header,512,768)
        cls.anchors={
            kind:{
                "trusted_eval_mod_65537":n.fingerprint(
                    n.shifted_future(cls.real_future,kind),n.P),
                "trusted_eval_mod_65521":n.fingerprint(
                    n.shifted_future(cls.real_future,kind),n.Q),
            } for kind in n.WORLD_KINDS
        }

    def test_fixed_moduli_and_exact_target_worlds(self):
        self.assertEqual((n.P,n.Q,n.R),(65537,65521,3))
        self.assertEqual(n.P*n.Q,4294049777)  # deterministic fixed offset
        self.assertEqual(n.SHIFT["product_shift"],n.P*n.Q)
        self.assertEqual(n.WORLD_KINDS,(
            "valid","unit_shift","p_shift","product_shift"))
        self.assertEqual(n.FRESH_OFFSET,94322000)
        self.assertEqual(n.SOURCE_COUNT,12)

    def test_modular_fingerprints_detect_unit_then_larger_collisions(self):
        target=self.real_future
        anchor=self.anchors
        self.assertNotEqual(anchor["valid"],anchor["unit_shift"])
        self.assertEqual(
            anchor["valid"]["trusted_eval_mod_65537"],
            anchor["p_shift"]["trusted_eval_mod_65537"])
        self.assertNotEqual(
            anchor["valid"]["trusted_eval_mod_65521"],
            anchor["p_shift"]["trusted_eval_mod_65521"])
        self.assertEqual(anchor["valid"],anchor["product_shift"])
        for modulus in (n.P,n.Q):
            d=target[-1]
            for kind in n.WORLD_KINDS:
                diff=(n.fingerprint(
                    n.shifted_future(target,kind),modulus)-
                    n.fingerprint(target,modulus))%modulus
                self.assertEqual(diff,
                    n.SHIFT[kind]*pow(n.R,len(target)-1,modulus)%modulus)

    def test_policies_never_see_world_kind_or_private_target(self):
        pub=self.public
        self.assertEqual(pub["predicted_output"],self.real_future)
        self.assertGreater(self.mask,0)
        self.assertEqual(pub["public_mask"],self.mask)
        altered=replace(self.case,program=None,
                        future=tuple(999 for _ in self.case.query))
        self.assertEqual(pub,n.select_public_candidate(
            altered,self.header,512,768))
        for kind,anchor in self.anchors.items():
            for arm in n.ARMS:
                got=n.choose_with_commitment(pub,anchor,arm)
                expected=n.EXACT_CORRECTNESS_MATRIX[kind][arm]
                self.assertEqual(got["accepted"],expected!="abstain")
                if arm in ("public_only","forged_pq"):
                    self.assertEqual(
                        got["extra_trusted_target_dependent_modular_values"],0)
                elif arm=="trusted_p":
                    self.assertEqual(
                        got["extra_trusted_target_dependent_modular_values"],1)
                else:
                    self.assertEqual(
                        got["extra_trusted_target_dependent_modular_values"],2)

    def test_sealed_native_grading_only_after_policy_freeze(self):
        row={
            "replicate":0,"observation_degree":512,"query_degree":768,
            "family":"wide_sparse_small","selected_attempt":self.attempt,
            "original_public_accept_mask":self.mask,
            "full_public_input_sha256":self.digest,
            "trusted_target_commitments_by_counterfactual_world":self.anchors,
        }
        proposal={**self.public,"world_choices":{
            kind:{arm:n.choose_with_commitment(
                self.public,self.anchors[kind],arm) for arm in n.ARMS}
            for kind in n.WORLD_KINDS}}
        got=n.postchoice_grader(row,proposal)
        self.assertEqual(set(got),set(n.WORLD_KINDS))
        self.assertEqual(got["valid"]["arms"]["trusted_pq"]["future_correct"],True)
        self.assertEqual(got["unit_shift"]["arms"]["trusted_pq"]["abstained"],True)
        self.assertEqual(got["p_shift"]["arms"]["trusted_p"]["false_accept"],True)
        self.assertEqual(
            got["product_shift"]["arms"]["trusted_pq"]["false_accept"],True)
        self.assertEqual(
            len({x["same_complete_public_input_sha256"]
                 for x in got.values()}),1)

    def test_public_source_identity_and_no_private_future_in_public_input(self):
        c=self.case
        self.assertEqual(c.future,())
        self.assertIsNone(c.program)
        self.assertEqual(self.digest,n.public_digest(c,512))
        self.assertEqual(self.planted[:4],(
            self.header.base,self.header.delay,
            self.header.spacing,self.header.amplitude))
        self.assertTrue(self.mask)
        self.assertEqual(len(c.query),769)

    def test_invalid_commitments_are_rejected(self):
        with self.assertRaises(ValueError):
            n.fingerprint((1,),313)
        with self.assertRaises(ValueError):
            n.shifted_future(self.real_future,"invalid")
        with self.assertRaises(ValueError):
            n.choose_with_commitment(self.public,{},"trusted_p")
        with self.assertRaises(ValueError):
            n.choose_with_commitment(self.public,{},"nonregistered")
        with self.assertRaises(ValueError):
            n.source_seed(7,512,"wide_sparse_small")


if __name__=="__main__":
    unittest.main()
