"""NOR-42 historical-only off-grammar suffix smoke, no new cohort generation."""
from __future__ import annotations
from itertools import combinations
from fractions import Fraction
import random
import unittest

from research.nor42_paired_program_ood import paired_model as x


class NOR42HistoricalSmokeTests(unittest.TestCase):
    def test_synthetic_future_first_difference_all_beta_and_ops(self):
        d=512
        q=3*d//2
        query=(1,)+tuple((i%5)-2 for i in range(1,q+1))
        clean=(1,)+tuple(2*i for i in range(1,q+1))
        for op in x.OPERATIONS:
            for beta in x.AMPLITUDES:
                hidden=x.hidden_future(query,clean,d,beta,op)
                self.assertEqual(hidden[:d+1],clean[:d+1])
                self.assertEqual(hidden[:d+32],clean[:d+32])
                self.assertEqual(hidden[d+32]-clean[d+32],beta)
                changed=x.discrepancy_support(clean,hidden)
                self.assertEqual(min(changed),d+32)
                self.assertGreater(len(changed),0)

    def test_exact_hypergeometric_probability_by_enumeration(self):
        m,k=8,3
        for s in range(m+1):
            support=set(range(s))
            total=sum(bool(support.intersection(plan))
                      for plan in combinations(range(m),k))
            total_plans=sum(1 for _ in combinations(range(m),k))
            self.assertEqual(x.uniform_query_hit_probability(m,s,k),
                             Fraction(total,total_plans))

    def test_oracle_vs_internal_model_consistency(self):
        d=512;q=3*d//2
        query=(1,)+tuple((i%7)-3 for i in range(1,q+1))
        clean=(1,)+tuple(0 for _ in range(q))
        hidden=x.hidden_future(query,clean,d,1,"I")
        before=tuple(range(16))
        containing=(d+32,)+tuple(range(15))
        self.assertFalse(x.oracle_detects_proposal(clean,hidden,before))
        self.assertTrue(x.oracle_detects_proposal(clean,hidden,containing))
        self.assertTrue(x.clean_internal_consistency(clean,clean,before))
        self.assertTrue(x.clean_internal_consistency(clean,clean,containing))
        self.assertTrue(x.oracle_detects_proposal(hidden,clean,containing))

    def test_only_sha_frozen_previous_nor40_source_ids(self):
        rows=x.original_history()
        self.assertEqual(len(rows),48)
        self.assertEqual({r["replicate"] for r in rows},set(range(76,84)))
        self.assertEqual({r["degree"] for r in rows},{512,1024})
        self.assertEqual({r["family"] for r in rows},
                         {"wide_dense_small","wide_sparse_small","wide_rational_short"})
        self.assertTrue(all(r["original_native_public_mask"].bit_count()==1
                            for r in rows))

    def test_full_48_archived_smoke_is_reproducible(self):
        report=x.historical_smoke()
        self.assertEqual(report["schema"],"nor42.archived_nor40_late_suffix_smoke.v1")
        self.assertEqual(report["new_nor42_source_units"],0)
        self.assertEqual(report["historical_source_units"],48)
        self.assertEqual(len(report["records"]),48)
        self.assertTrue(all(r["paired_public_prefix_equal"] and
                            not r["paired_future_targets_equal"]
                            for r in report["records"]))
        self.assertTrue(all(r["first_difference_index"]==r["last_public_index"]+32
                            for r in report["records"]))
        self.assertEqual(x.canonical(report),x.canonical(x.historical_smoke()))


if __name__=="__main__":
    unittest.main()
