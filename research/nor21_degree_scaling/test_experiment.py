"""NOR-21 dynamic public horizon, private-target boundary and CPU tests."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import random
import unittest

from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as n


class NOR21ScaledAndOutOfGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good=n.sample_source(0,512,"wide_dense_small","in_grammar")
        cls.bad=n.sample_source(0,512,"wide_dense_small","public_corrupt")
        cls.shift=n.sample_source(0,512,"wide_dense_small","future_shift")

    def test_frozen_source_grid_and_disjoint_public_hashes(self):
        self.assertEqual(n.DEGREES,(512,1024))
        self.assertEqual(n.TRIALS,3)
        self.assertEqual(n.WARMUPS,1)
        self.assertEqual(n.REPLICATES,2)
        self.assertEqual(n.SAMPLE_COUNT,36)
        self.assertEqual(n.SEED_OFFSET,94210000)
        self.assertEqual(n.seed(0,512,"wide_dense_small","in_grammar"),
                         corpus.SEED+94210000)
        digests=[x[-1] for x in (self.good,self.bad,self.shift)]
        self.assertEqual(len(set(digests)),3)

    def test_exact_scaled_masks_agree_for_all_four_algorithms(self):
        for case,h,prog,attempt,mask,unique,digest in (
                self.good,self.bad,self.shift):
            self.assertEqual(n.public_digest(case,512),digest)
            self.assertEqual(n.native_mask(case,h,512),mask)
            reference=(mask,unique)
            for mode in n.MODES:
                self.assertEqual(n.public_verifier(case,h,512,mode),mask)
                v=n.certify(case,h,512,768,mode)
                self.assertEqual((v["mask"],v["unanimous"]),reference)

    def test_out_of_grammar_public_demo_corruption_must_abstain(self):
        c,h,p,attempt,mask,unanimous,digest=self.bad
        self.assertEqual(mask,0)
        self.assertFalse(unanimous)
        for mode in n.MODES:
            ans=n.certify(c,h,512,768,mode)
            self.assertIsNone(ans["chosen_class"])
            self.assertFalse(ans["unanimous"])
            self.assertEqual(ans["status"],"no_public_candidate_abstain")

    def test_unobservable_future_override_fails_family_conditional_truth(self):
        c,h,p,attempt,mask,unique,digest=self.shift
        self.assertTrue(unique)
        # The policy returns the same result when hidden truth / program
        # fields are replaced, because both fields are unavailable.
        alt=replace(c,program=("R",),future=(999,))
        for mode in n.MODES:
            self.assertEqual(n.certify(c,h,512,768,mode),
                             n.certify(alt,h,512,768,mode))
        public_choice=n.certify(c,h,512,768,"stream16")
        record=self.to_fixture_entry(self.shift,"future_shift")
        graded=n.graded_after_freeze(record,public_choice)
        self.assertTrue(graded["false_accept_outside_family"])
        self.assertFalse(graded["future_correct"])
        witness=graded["counterfactual_public_same_future_different"]
        self.assertEqual(witness["same_public_sha256"],digest)
        self.assertNotEqual(witness["valid_world_future_sha256"],
                            witness["mutant_world_future_sha256"])

    @staticmethod
    def to_fixture_entry(result,kind):
        c,h,p,attempt,mask,unique,digest=result
        degree=512
        return {
            "replicate":0,"observation_degree":degree,"query_degree":768,
            "family":c.family,"case_kind":kind,
            "selected_attempt":attempt,
            "exact_public_accept_mask":mask,
            "hypothetical_query_future_unanimous":unique,
            "full_public_input_sha256":digest,
        }

    def test_in_grammar_certificate_grades_true_after_freeze(self):
        c,h,p,attempt,mask,unique,digest=self.good
        self.assertTrue(unique)
        selection=n.certify(c,h,512,768,"stream16")
        self.assertIsNotNone(selection["chosen_class"])
        graded=n.graded_after_freeze(
            self.to_fixture_entry(self.good,"in_grammar"),selection)
        self.assertTrue(graded["future_correct"])
        self.assertFalse(graded["false_accept_outside_family"])
        self.assertIsNone(graded["counterfactual_public_same_future_different"])

    def test_1024_native_and_streaming_unit_series_semantics(self):
        # Lightweight input with a genuine reciprocal base at n=1024
        # checks dynamic bounds without allocating a full new fixture.
        rng=random.Random(21)
        f=(1,)+tuple((rng.choice((-1,1)) if i%59==0 else 0)
                     for i in range(1,1025))
        p=("R",64,24,1,"N","S")
        y=corpus.execute(f,p)
        h=corpus.Header(p[0],p[1],p[2],p[3])
        c=corpus.Case("nor21-1024-unit","sparse",
            ((f,y),(f,y)),f,(),None)
        self.assertEqual(n.native_mask(c,h,1024),
                         n.stream_mask(c,h,1024))
        self.assertEqual(n.cached_mask(c,h,1024),
                         n.chunk4_mask(c,h,1024))

    def test_timed_costs_and_memory_follow_public_code_paths(self):
        c,h,_,_,mask,_,_=self.good
        for mode in n.MODES:
            result=n.certify(c,h,512,768,mode,measure=True)
            t=result["times"]
            self.assertGreater(t["query_ns"],0)
            self.assertGreater(t["verification_ns"],0)
            self.assertEqual(t["total_ns"],
                             t["query_ns"]+t["verification_ns"]+
                             t["certificate_ns"])
            peak=n.allocation(c,h,512,768,mode)
            self.assertEqual(peak["mask"],mask)
            self.assertGreater(peak["peak_bytes"],0)
            self.assertGreaterEqual(peak["peak_bytes"],peak["current_bytes"])

    def test_graded_world_and_invalid_source_are_independent(self):
        bad=self.to_fixture_entry(self.bad,"public_corrupt")
        c,h,p,attempt,mask,unique,digest=self.bad
        choice=n.certify(c,h,512,768,"stream16")
        result=n.graded_after_freeze(bad,choice)
        self.assertTrue(result["public_no_match_abstention"])
        self.assertFalse(result["false_accept_outside_family"])
        self.assertFalse(result["public_certificate_issued"])

    def test_sample_bounds_and_invalid_policy_mode_fail_closed(self):
        c,h,*_=self.good
        with self.assertRaises(ValueError):
            n.seed(3,512,"wide_dense_small","in_grammar")
        with self.assertRaises(ValueError):
            n.public_source(random.Random(0),"bad",512)
        with self.assertRaises(ValueError):
            n.public_verifier(c,h,512,"invalid")
        with self.assertRaises(ValueError):
            n.certify(c,h,512,768,"invalid")


if __name__=="__main__":
    unittest.main()
