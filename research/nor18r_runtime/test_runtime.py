"""NOR-18R tests: public-source parity, reciprocal operands and CPU accounting."""
from __future__ import annotations
from dataclasses import replace
from collections import Counter
import json
import math
import random
import statistics
import unittest

from research.nor18r_runtime import benchmark as r
from research.rrc_compactification import compound25 as corpus


class NOR18RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=random.Random(202611181816)
        cases=[]
        for family in ("late_source","signed","rational"):
            for t in range(100):
                case=corpus.make_case(rng,family,f"nor18r-parity-{family}-{t}")
                heads=corpus.possible_headers(case)
                if len(heads)==1:
                    cases.append((case,heads[0]))
                    break
            else:
                raise AssertionError("could not generate positive unique public header")
        cls.cases=cases

    def test_exact_public_full_masks_and_prediction_certificates_equal(self):
        for case,header in self.cases:
            expected=r.mask_independent(case,header)
            self.assertGreater(expected,0)
            for mode in r.MODES:
                self.assertEqual(r.evaluated_full_mask(case,header,mode),expected)
                out=r.outcome(case,header,mode)
                self.assertEqual(out["mask"],expected)
                self.assertEqual(out["unanimous"],r.outcome(
                    case,header,"independent16")["unanimous"])

    def test_sealed_future_and_program_identity_do_not_change_algorithms(self):
        for c,h in self.cases:
            contaminated=replace(c,program=None,
                future=tuple(123456789 for _ in c.future))
            for mode in r.MODES:
                self.assertEqual(r.outcome(c,h,mode),
                                 r.outcome(contaminated,h,mode))

    def test_public_exact_inverse_recursion_product_counts(self):
        for c,h in self.cases:
            for f,y in c.demos:
                inv=r.exact_inverse_operand_ledger(f[:193])
                self.assertEqual(inv["multiplications"],192*193//2)
                self.assertGreater(inv["operand_bitwidth_products"],0)
                self.assertGreaterEqual(inv["max_operand_bits"],1)
                self.assertGreaterEqual(inv["max_inverse_coefficient_bits"],1)
                self.assertGreater(inv["max_inverse_coefficient_python_int_bytes"],0)
            query=r.exact_inverse_operand_ledger(c.query)
            self.assertEqual(query["multiplications"],256*257//2)
            assert query["operand_bitwidth_products"]>0

    def test_per_mode_reciprocal_count_and_width_ledger(self):
        for c,h in self.cases:
            short={str(d):r.nor17.prefix_mask(c,h,d) for d in (160,176)}
            x=r.integer_cost_ledger(c,h,short)
            self.assertEqual(set(x["per_mode"]),set(r.MODES))
            self.assertEqual(x["per_mode"]["cached16"]["R_invocations_full_public_demos"],2)
            self.assertEqual(x["per_mode"]["independent16"]["R_invocations_full_public_demos"],
                             2*(16*int(h.base=="R")+8))
            self.assertEqual(x["per_mode"]["cached16"]["R_invocations_query"],1)
            self.assertGreater(
                x["per_mode"]["independent16"]["R_operand_bitwidth_product_proxy"],
                x["per_mode"]["cached16"]["R_operand_bitwidth_product_proxy"])
            self.assertEqual(
                x["per_mode"]["independent16"]["R_integer_multiplications_executed"]-
                x["per_mode"]["cached16"]["R_integer_multiplications_executed"],
                2*((16*int(h.base=="R")+8)-1)*192*193//2)

    def test_peak_memory_is_untimed_and_matches_public_mask(self):
        c,h=self.cases[0]
        for mode in r.MODES:
            z=r.ephemeral_memory_profile(c,h,mode)
            self.assertEqual(z["mask"],r.mask_independent(c,h))
            self.assertGreaterEqual(z["python_tracemalloc_peak_bytes"],
                                    z["python_tracemalloc_current_bytes"])
            self.assertGreater(z["python_tracemalloc_peak_bytes"],0)

    def test_full_benchmark_schema_and_bootstrap(self):
        cases=[]
        for cohort in range(4):
            for i in range(16):
                cases.append({
                    "cohort":cohort,
                    "speedup_independent_over_cached_total":2.0+cohort*.1,
                })
        b=r.block_bootstrap(cases,draws=128)
        self.assertEqual(b["draws"],128)
        self.assertLessEqual(b["percentile_2_5"],b["percentile_97_5"])
        self.assertTrue(2<=b["percentile_2_5"]<=b["percentile_97_5"]<=2.3)
        self.assertEqual(len(r.MODES),4)
        self.assertEqual(r.SOURCES,64)
        self.assertEqual(r.TIMED_REPS,7)
        self.assertEqual(r.WARMUPS,2)
        self.assertEqual(r.BOOTSTRAPS,4096)

    def test_ledger_has_zero_unpaid_future_access(self):
        c,h=self.cases[0]
        masked=corpus.Case(c.identifier,c.family,c.demos,c.query,(),None)
        for mode in r.MODES:
            self.assertEqual(r.outcome(c,h,mode),r.outcome(masked,h,mode))
        self.assertEqual(r.QUERY_DEGREE,256)
        self.assertEqual(r.PUBLIC_DEGREE,192)


if __name__=="__main__":
    unittest.main()
