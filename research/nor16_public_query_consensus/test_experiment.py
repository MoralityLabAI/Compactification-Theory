"""NOR-16 public feature / private truth separation and exact algebraic parity."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import random
import unittest

from research.nor16_public_query_consensus import experiment as n
from research.rrc_compactification import compound25 as corpus


class NOR16PublicConsensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=random.Random(2026111616)
        for trial in range(120):
            case=corpus.make_case(
                rng,"late_source",f"nor16-unit-{trial:05d}")
            h=corpus.possible_headers(case)
            if len(h)==1:
                cls.case=case
                cls.header=h[0]
                break
        else:
            raise AssertionError("could not sample positive unique public header")

    def test_query_predictions_match_exact_source_interpreter_for_all_classes(self):
        predictions=n.public_candidate_predictions(self.case,self.header)
        self.assertEqual(len(predictions),16)
        for j,one in enumerate(predictions):
            self.assertEqual(one,corpus.execute(
                self.case.query,n.program_for(self.header,j)))
        with self.assertRaises(ValueError):
            n.program_for(self.header,16)

    def test_public_candidate_simulations_ignore_hidden_program_and_future(self):
        altered=replace(self.case,
            program=None,future=tuple(7654321 for _ in self.case.future))
        p=n.public_candidate_predictions(self.case,self.header)
        self.assertEqual(p,n.public_candidate_predictions(altered,self.header))
        self.assertEqual(n.exact_public_mask(self.case,self.header),
                         n.exact_public_mask(altered,self.header))
        self.assertEqual(n.full_public_digest(self.case),
                         n.full_public_digest(altered))
        self.assertEqual(n.visible_source_digest(self.case),
                         n.visible_source_digest(altered))
        self.assertEqual(n.hash_order(self.case),n.hash_order(altered))
        self.assertEqual(n.group_candidates(p),n.group_candidates(
            n.public_candidate_predictions(altered,self.header)))

    def test_exact_map_is_conditional_candidate_output_multiplicity(self):
        predictions=[(i,) for i in range(16)]
        predictions[0]=predictions[1]=predictions[2]=(999,)
        predictions[4]=predictions[5]=(555,)
        original=list(range(15,-1,-1))
        # Three accepted candidates agree, two agree on another output.
        mask=sum(1<<j for j in (0,1,2,4,5,6))
        best,mass,distinct=n.mask_map_choice(mask,predictions,original)
        self.assertEqual(mass,3)
        self.assertEqual(best,2)
        self.assertEqual(distinct,3)
        with self.assertRaises(ValueError):
            n.mask_map_choice(0,predictions,original)

    def test_public_counterfactual_certificate_matches_equal_visible_demos(self):
        predictions=n.public_candidate_predictions(self.case,self.header)
        accepted=n.exact_public_mask(self.case,self.header)
        pair=n.counterfactual_pair(accepted,predictions)
        if pair is not None:
            j,k=pair["candidate_classes"]
            self.assertTrue(accepted&(1<<j))
            self.assertTrue(accepted&(1<<k))
            self.assertNotEqual(predictions[j],predictions[k])
            idx=pair["first_divergent_future_output_degree"]
            self.assertEqual(predictions[j][:idx],predictions[k][:idx])
            self.assertNotEqual(predictions[j][idx],predictions[k][idx])
            assert all(corpus.verify_template(self.case,self.header,
                      corpus.OPS[c//4],corpus.OPS[c%4]) for c in (j,k))
            def reconstruct_valid_world(which):
                program=n.program_for(self.header,which)
                demos=tuple((source,corpus.execute(source,program))
                            for source,target in self.case.demos)
                return replace(self.case,program=program,
                               demos=demos,
                               future=corpus.execute(self.case.query,program))
            altA=reconstruct_valid_world(j)
            altB=reconstruct_valid_world(k)
            # Both worlds satisfy the ENTIRE generative program equation,
            # including hidden demo suffixes, with identical PUBLIC input.
            self.assertEqual(altA.visible,altB.visible)
            self.assertEqual(altA.visible,self.case.visible)
            self.assertEqual(altA.query,altB.query)
            self.assertEqual(altA.future,predictions[j])
            self.assertEqual(altB.future,predictions[k])
            self.assertNotEqual(altA.future,altB.future)
            self.assertEqual(n.full_public_digest(altA),n.full_public_digest(altB))
            self.assertEqual(corpus.grade(altA,altA.program)["future_correct"],1)
            self.assertEqual(corpus.grade(altB,altB.program)["future_correct"],1)

    def test_four_check_cost_and_sealed_future_grader(self):
        ck,_,_=n.nor11.train_source()
        a=n.single_case(self.case,self.header,ck)
        changed=replace(self.case,program=None,
            future=tuple(99999 for _ in self.case.future))
        b=n.single_case(changed,self.header,ck)
        for name in n.ARMS:
            self.assertEqual(a["arms"][name]["chosen_class"],
                             b["arms"][name]["chosen_class"])
            self.assertEqual(a["arms"][name]["verification_trace"],
                             b["arms"][name]["verification_trace"])
            self.assertEqual(a["arms"][name]["public_checks"],
                             b["arms"][name]["public_checks"])
            if name.endswith("_4"):
                self.assertLessEqual(a["arms"][name]["public_checks"],4)
            else:
                self.assertEqual(a["arms"][name]["public_checks"],16)
            self.assertEqual(a["arms"][name]["query_prediction_candidates"],
                             0 if name=="original_4" else 16)
        self.assertEqual(a["public_accept_mask"],b["public_accept_mask"])
        self.assertEqual(a["candidate_public_prediction_sha256"],
                         b["candidate_public_prediction_sha256"])
        self.assertEqual(a["full_public_input_sha256"],
                         b["full_public_input_sha256"])
        self.assertEqual(a["family"],"late_source")
        self.assertEqual(len(a["candidate_public_prediction_sha256"]),16)
        self.assertEqual(a["uniform_accepted_candidate_MAP_future_probability_den"],
                         a["public_accept_mask"].bit_count())

    def test_fresh_budget_cannot_expand(self):
        with self.assertRaises(ValueError):
            n.fresh_cohorts({},reps=5)
        with self.assertRaises(ValueError):
            n.fresh_cohorts({},late_per=49)
        with self.assertRaises(ValueError):
            n.fresh_cohorts({},signed_per=13)
        with self.assertRaises(ValueError):
            n.fresh_cohorts({},rational_per=13)

    def test_primary_pairing_checks_case_outcomes(self):
        records=[]
        for i,(b,c) in enumerate([(0,1),(1,0),(0,0),(1,1)]):
            records.append({"family":"late_source",
                            "arms":{
                                "original_4":{"grade":{"future_correct":b}},
                                "consensus_4":{"grade":{"future_correct":c}}}})
        p=n.paired(records,"original_4","consensus_4")
        self.assertEqual(p["paired_gains"],1)
        self.assertEqual(p["paired_losses"],1)
        self.assertEqual(p["net"],0)
        self.assertEqual(p["primary_distinct_public_inputs"],4)
        self.assertEqual(p["two_sided_exact_case_sign_p"],1.)


if __name__=="__main__":
    unittest.main()
