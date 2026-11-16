"""NOR-17: public-only prefix soundness, exact costs and selective proof tests."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch
import json
import random
import unittest

from research.nor17_prefix_certificates import experiment as n
from research.rrc_compactification import compound25 as corpus,model25


class NOR17ExactVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,_,_=n.nor11.train_source()
        cls.ck=ck
        rng=random.Random(20261116017)
        for i in range(120):
            case=corpus.make_case(rng,"late_source",f"nor17-tests-{i}")
            heads=corpus.possible_headers(case)
            if len(heads)==1:
                cls.case=case
                cls.header=heads[0]
                break
        else:
            raise AssertionError("no unique public header")

    def test_public_query_candidate_outputs_agree_with_native_interpreter(self):
        predicted=n.candidate_outputs(self.case.query,self.header,corpus.FUTURE)
        self.assertEqual(len(predicted),16)
        for j,output in enumerate(predicted):
            self.assertEqual(output,corpus.execute(
                self.case.query,n.program(self.header,j)))
        self.assertEqual(len(predicted[0]),257)
        with self.assertRaises(ValueError):
            n.candidate_outputs(self.case.query,self.header,97)

    def test_short_prefix_rejection_is_exact_and_superset_of_full_accepted(self):
        observed=n.prefix_mask(self.case,self.header)
        full=0
        for j in range(16):
            exact=n.accepted_full(self.case,self.header,j)
            compatible=all(corpus.execute(
                source[:n.PREFIX_DEGREE+1],n.program(self.header,j))
                ==target[:n.PREFIX_DEGREE+1]
                for source,target in self.case.demos)
            self.assertEqual(compatible,bool(observed&(1<<j)))
            if exact:
                full|=1<<j
                self.assertTrue(observed&(1<<j))
        self.assertNotEqual(full,0)
        self.assertEqual(full & observed,full)
        self.assertGreaterEqual(observed.bit_count(),full.bit_count())

    def test_sealed_future_and_program_cannot_change_any_policy_decision(self):
        predictions,prefix,order,policies=n.policies_before_grading(
            self.case,self.header,self.ck)
        changed=replace(self.case,program=None,
            future=tuple(987654321 for _ in self.case.future))
        other,prefix2,order2,second=n.policies_before_grading(
            changed,self.header,self.ck)
        self.assertEqual(predictions,other)
        self.assertEqual(prefix,prefix2)
        self.assertEqual(order,order2)
        self.assertEqual(policies,second)
        self.assertEqual(set(policies),set(n.KINDS))
        for kind,result in policies.items():
            limit=int(kind.rsplit("_",1)[-1])
            self.assertLessEqual(result["public_checks"],limit)
            if kind.startswith("greedy"):
                self.assertEqual(result["prefix_checks"],0)
                self.assertEqual(result["candidate_query_compositions"],0)
            else:
                self.assertEqual(result["candidate_query_compositions"],16)
                self.assertEqual(result["query_unary_transformations"],4)
                if kind.startswith("prefix"):
                    self.assertEqual(result["prefix_checks"],16)
                    self.assertEqual(result["prefix_demo_compositions"],32)
                    self.assertEqual(result["prefix_unary_transformations"],8)
                else:
                    self.assertEqual(result["prefix_checks"],0)

    def test_public_prediction_confirmation_requires_all_possible_futures(self):
        # All first eight classes predict A and the rest B.
        future=[("A",)]*8+[("B",)]*8
        valid={0}
        with patch.object(n,"accepted_full",side_effect=lambda c,h,j:j in valid):
            filtered=n.certifier(None,None,future,[0,1],4,0b11,use_prefix=True)
            self.assertTrue(filtered["certified"])
            self.assertEqual(filtered["chosen_class"],0)
            self.assertEqual(filtered["public_checks"],1)
            self.assertTrue(filtered["full_output_unanimity"])
            without=n.certifier(None,None,future,list(range(16)),4,
                                (1<<16)-1,use_prefix=False)
            self.assertFalse(without["certified"])
            self.assertIsNone(without["chosen_class"])
            self.assertEqual(without["public_checks"],4)
            self.assertFalse(without["full_output_unanimity"])

    def test_two_proven_positive_candidates_with_distinct_outputs_force_abstention(self):
        future=[("A",)]*8+[("B",)]*8
        full=(1<<0)|(1<<8)
        with patch.object(n,"accepted_full",side_effect=lambda c,h,j:j in (0,8)):
            out=n.certifier(None,None,future,[0,8],4,full,use_prefix=True)
        self.assertFalse(out["certified"])
        self.assertIsNone(out["chosen_class"])
        self.assertEqual(out["public_checks"],2)
        self.assertEqual(out["exit"],"incompatible_full_accepted_outputs")
        self.assertEqual(out["confirmed_accepted_classes"],[0,8])
        with self.assertRaises(ValueError):
            n.certifier(None,None,future,[0],4,full,True)

    def test_single_case_full_retrospective_mask_and_sound_certification(self):
        out=n.single_case(self.case,self.header,self.ck)
        self.assertEqual(out["family"],"late_source")
        self.assertEqual(set(out["arms"]),set(n.KINDS))
        self.assertEqual(out["retrospective_public_full_checks"],16)
        full=out["full_public_accept_mask_retrospective"]
        partial=out["short_prefix_survivor_mask"]
        self.assertEqual(full&partial,full)
        self.assertTrue(full)
        for kind,arm in out["arms"].items():
            self.assertEqual(arm["grade"]["accepted"],
                arm["grade"]["future_correct"]+arm["grade"]["false_accept"])
            if arm["certified"]:
                self.assertEqual(arm["grade"]["future_correct"],1)
                self.assertEqual(arm["grade"]["false_accept"],0)
                self.assertTrue(arm["full_output_unanimity"])
        self.assertEqual(len(out["hypothetical_query_output_hashes"]),16)

    def test_fresh_sampling_budget_bounds(self):
        for kwargs in ({"replicates":5},{"late_per":49},
                       {"signed_per":13},{"rational_per":13}):
            with self.assertRaises(ValueError):
                n.frozen_cohorts({},**kwargs)
        with self.assertRaises(ValueError):
            n.policy_greedy(None,None,[],3)
        with self.assertRaises(ValueError):
            n.prefix_mask(self.case,self.header,degree=100)


if __name__=="__main__":
    unittest.main()
