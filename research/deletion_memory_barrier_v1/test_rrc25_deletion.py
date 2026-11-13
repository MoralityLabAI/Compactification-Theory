"""DMB-01B source-locked learned ranker diagnostic tests."""
from __future__ import annotations
import math
import random
import unittest
from dataclasses import replace
from fractions import Fraction as Q

from bench_rrc25_deletion import (
    STATUS, empirical_prior, full_one_trace_success,
    one_prior_report, ranked_word, run,
)
from reference import (
    finite_prior_ceiling, optimal_terminal_success_weighted,
    streaming_laws_for_words, words,
)
from research.rrc_compactification import bench25, compound25


class StructuredSourceBounds(unittest.TestCase):
    def test_nonuniform_terminal_prior_topK_bound(self):
        prior={"0000":Q(3,5),"1111":Q(1,5),
               "1100":Q(1,10),"0011":Q(1,10)}
        self.assertEqual(finite_prior_ceiling(prior,0),Q(3,5))
        self.assertEqual(finite_prior_ceiling(prior,1),Q(4,5))
        self.assertEqual(finite_prior_ceiling(prior,2),Q(1))
        self.assertEqual(finite_prior_ceiling(prior,0,1,clock=True),Q(4,5))
        with self.assertRaises(ValueError):
            finite_prior_ceiling({"0":Q(1,3)},1)

    def test_weighted_bayes_terminal_state_respects_prior(self):
        prior={"00":Q(3,5),"11":Q(1,4),"01":Q(3,20)}
        for q in (Q(1,2),Q(4,5)):
            for m in range(4):
                for B in (1,2):
                    laws=streaming_laws_for_words(prior,q,m,B)
                    success=optimal_terminal_success_weighted(laws,prior)
                    self.assertGreaterEqual(success,Q(3,5))
                    self.assertLessEqual(success,finite_prior_ceiling(prior,B))
        self.assertEqual(full_one_trace_success(prior,Q(1)),Q(3,5))

    def test_empirical_multiplicities_preserved(self):
        records=[{"ranked_trace_word":"10101010"}]*3 + [
            {"ranked_trace_word":"00001111"}]
        prior=empirical_prior(records)
        self.assertEqual(prior,{"00001111":Q(1,4),"10101010":Q(3,4)})
        self.assertEqual(finite_prior_ceiling(prior,0),Q(3,4))


class TrainedRankerSourceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoints,hashes=bench25.checkpoints()
        cls.checkpoint=checkpoints["recurrent"]
        cls.hash=hashes["recurrent"]
        cls.case=compound25.make_case(random.Random(940200),"signed",
                                      "dmb01b-source-lock")

    def test_original_checkpoint_source_hashes(self):
        self.assertEqual(self.checkpoint["kind"],"recurrent")
        self.assertEqual(self.checkpoint["parameters"],1000)
        self.assertEqual(len(self.hash),64)

    def test_ranker_output_independent_of_hidden_future(self):
        original=ranked_word(self.case,self.checkpoint)
        mutated=replace(self.case,future=tuple(
            ((i%3)-1)*10**8 for i in range(len(self.case.future))))
        altered=ranked_word(mutated,self.checkpoint)
        self.assertEqual(original,altered)
        self.assertEqual(len(original["ranked_trace_word"]),8)
        self.assertEqual(len(set(original["ranked_class_ids"])),2)

    def test_small_source_locked_monitor_run(self):
        report=run(test_per=1,deletions=(Q(1,2),),
                   counts=(0,1,2),bits=(2,4))
        self.assertEqual(report["status"],STATUS)
        self.assertEqual(report["checkpoint_sha256"],self.hash)
        self.assertEqual(report["model_parameters"],1000)
        self.assertEqual(report["source_cases"],6)
        self.assertLessEqual(report["source_prior_support"],6)
        self.assertTrue(report["native_model_rankings_not_hidden_cot"])
        self.assertTrue(report["posterior_table_from_same_finite_prior"])
        self.assertFalse(report["external_theorem_122_140_reproduced"])
        for row in report["results"][0]["monitor_grid"]:
            self.assertLessEqual(Q(row["success"]),
                                 Q(row["prior_topK_ceiling"]))
        self.assertTrue(math.isfinite(report["source_prior_shannon_entropy_bits"]))

    def test_reject_invalid_case_grid(self):
        with self.assertRaises(ValueError):
            run(test_per=0)
        with self.assertRaises(ValueError):
            one_prior_report({"00":Q(1)},Q(1,2),counts=(-1,),bits=(1,))


if __name__=="__main__":
    unittest.main()
