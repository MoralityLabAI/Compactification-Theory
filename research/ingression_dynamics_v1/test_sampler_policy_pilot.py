"""QO-08 pilot reproducibility, pairing, provenance and output-contract checks."""
import json
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path

from research.ingression_dynamics_v1.sampler_policy_pilot import (
    ALPHA, BUDGET, SEEDS, POLICIES, FAMILIES, run_pilot,
)


class SamplerPilotTests(unittest.TestCase):
    def test_frozen_protocol_matrix(self):
        self.assertEqual(SEEDS,tuple(range(101,113)))
        self.assertEqual(BUDGET,96)
        self.assertEqual(ALPHA,Q(1,20))
        self.assertEqual(POLICIES,("round_robin","width","bottleneck"))
        self.assertEqual(len(FAMILIES),2)

    def test_paired_development_receipts_are_deterministic(self):
        first=run_pilot(SEEDS[:2],12)
        second=run_pilot(SEEDS[:2],12)
        self.assertEqual(first,second)
        self.assertEqual(first["protocol"]["seeds"],[101,102])
        self.assertEqual(first["protocol"]["budget_per_policy"],12)
        self.assertEqual(len(first["records"]),2*2*3)
        self.assertEqual(len(first["summaries"]),2*3)
        self.assertEqual(len(first["comparisons"]),2*2)
        expected_keys={(family,seed,policy)
                       for family,_,_ in FAMILIES for seed in SEEDS[:2]
                       for policy in POLICIES}
        self.assertEqual({(r["family"],r["seed"],r["policy"])
                          for r in first["records"]},expected_keys)
        for r in first["records"]:
            self.assertEqual(r["observed_queries"],12)
            self.assertEqual(sum(sum(row) for row in r["query_counts"]),12)
            self.assertEqual(len(r["receipt_hash"]),64)
            if r["covered_true_rates"]:
                self.assertFalse(r["false_certificate_on_truth"])
                self.assertLessEqual(Q(r["true_frozen_error"]),
                                     Q(r["robust_upper"]))
        for compare in first["comparisons"]:
            self.assertEqual(compare["candidate_better_seed_count"]+
                compare["candidate_equal_seed_count"]+
                compare["candidate_worse_seed_count"],2)
        self.assertEqual(len(first["source_sha256"]),3)
        self.assertTrue(all(len(v)==64 for v in first["source_sha256"].values()))
        self.assertEqual(first,json.loads(json.dumps(first)))

    def test_seed_and_budget_validation(self):
        for seeds, budget in (((1,1),10), ((),10), ((1,),-1),
                              ((True,),10)):
            with self.assertRaises(ValueError):
                run_pilot(seeds,budget)


if __name__ == "__main__":
    unittest.main()
