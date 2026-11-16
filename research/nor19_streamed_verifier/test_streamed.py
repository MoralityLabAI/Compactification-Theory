"""NOR-19 semantic, buffer and source-isolation tests.

No benchmark timing result is exposed to the selection logic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random
import statistics
import unittest

from research.nor19_streamed_verifier import benchmark as n
from research.rrc_compactification import compound25 as corpus


class NOR19StreamedCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=random.Random(202611191916)
        cls.examples=[]
        for family in ("late_source","signed","rational","product","sparse","squared"):
            for trial in range(200):
                c=corpus.make_case(rng,family,f"nor19-unit-{family}-{trial}")
                heads=corpus.possible_headers(c)
                if len(heads)==1:
                    cls.examples.append((c,heads[0]))
                    break
            else:
                raise AssertionError("could not find uniquely identified source header")
        assert len(cls.examples)==6

    def test_all_five_exact_public_masks_and_certificates_match(self):
        for case,h in self.examples:
            ref=n.public_mask(case,h,"independent16")
            self.assertGreater(ref,0)
            for mode in n.SOURCE_MODELS:
                self.assertEqual(n.public_mask(case,h,mode),ref)
                receipt=n.complete(case,h,mode)
                self.assertEqual(receipt["mask"],ref)
                self.assertEqual(receipt["unanimous"],
                    n.complete(case,h,"cached16")["unanimous"])
                self.assertEqual(receipt["chosen_class"],
                    n.complete(case,h,"cached16")["chosen_class"])

    def test_one_complete_candidate_matches_original_exact_interpreter(self):
        for case,h in self.examples:
            for source,observed in case.demos:
                source=source[:corpus.OBSERVED+1]
                unary={op:corpus.d.operate(source,op) for op in corpus.OPS}
                for j in range(16):
                    expected=corpus.execute(source,n.old.nor17.program(h,j))
                    got=n.build_one_candidate(h,unary,j)
                    self.assertEqual(got,expected)
                with self.assertRaises(ValueError):
                    # Candidate source indexing is exact: no invented class.
                    n.public_mask(case,h,"unknown")

    def test_no_private_source_program_or_future_observed_by_any_mode(self):
        for case,h in self.examples:
            altered=replace(case,program=None,
                            future=tuple(17_000_001 for _ in case.future))
            for mode in n.SOURCE_MODELS:
                self.assertEqual(n.complete(case,h,mode),
                                 n.complete(altered,h,mode))
        self.assertEqual(n.SOURCES,64)

    def test_staged_timing_is_accounted_for_both_execution_orders(self):
        case,h=self.examples[0]
        for mode in n.SOURCE_MODELS:
            z=n.complete(case,h,mode,measure=True)
            values=z["times"]
            self.assertEqual(values["total_ns"],
                values["query_ns"]+values["verification_ns"]+
                values["certificate_ns"])
            self.assertGreater(values["verification_ns"],0)
            self.assertGreater(values["query_ns"],0)
            self.assertGreater(values["total_ns"],0)

    def test_streamed_and_chunked_peak_memory_measured_separately(self):
        case,h=self.examples[0]
        for mode in n.SOURCE_MODELS:
            out=n.allocation_profile(case,h,mode)
            ver=n.verifier_only_profile(case,h,mode)
            self.assertEqual(out["mask"],ver["mask"])
            self.assertLessEqual(out["python_current_allocation_bytes"],
                                 out["python_peak_allocation_bytes"])
            self.assertGreater(out["python_peak_allocation_bytes"],0)
            self.assertGreater(ver["python_verifier_only_peak_bytes"],0)

    def test_bigint_operation_count_is_exact_for_streamed_cache(self):
        for case,h in self.examples:
            original=n.observed_bigint_work(case,h,"independent16")
            cached=n.observed_bigint_work(case,h,"cached16")
            self.assertEqual(n.inverse_invocations(h,"independent16"),
                             8+16*int(h.base=="R"))
            for mode in ("cached16","stream16_queryfirst",
                         "stream16_verifyfirst","chunk4_verifyfirst"):
                self.assertEqual(n.observed_bigint_work(case,h,mode),cached)
                self.assertEqual(n.inverse_invocations(h,mode),1)
            self.assertGreater(original["R_integer_products"],
                               cached["R_integer_products"])
            self.assertGreater(original["R_operand_bitwidth_product_proxy"],
                               cached["R_operand_bitwidth_product_proxy"])

    def test_source_selection_frozen_no_overlap(self):
        indexes=[i for i in range(8,16)]
        indexes+=list(range(52,56))+list(range(64,68))
        old=list(range(8))+list(range(48,52))+list(range(60,64))
        self.assertEqual(len(indexes),len(old))
        self.assertFalse(set(indexes)&set(old))
        self.assertEqual(n.TIMED_REPS,7)
        self.assertEqual(n.WARMUPS,2)
        self.assertEqual(n.PRIMARY_MEMORY_RATIO_MAX,.80)
        self.assertEqual(n.PRIMARY_SPEEDUP_NATIVE_MIN,2.0)
        self.assertEqual(n.PRIMARY_MEMORY_WINS_MIN,48)

    def test_four_cohort_block_bootstrap(self):
        records=[{"cohort":i,"primary_peak_ratio_vs_cached":.5+.1*i,
                  "primary_native_speedup":3+i*.25}
                 for i in range(4) for _ in range(16)]
        a=n.bootstrap_ratio(records,"primary_peak_ratio_vs_cached",draws=80)
        b=n.bootstrap_ratio(records,"primary_native_speedup",draws=80)
        for r in (a,b):
            self.assertEqual(r["draws"],80)
            self.assertLessEqual(r["percentile_2_5"],r["percentile_97_5"])
        self.assertEqual(len(n.SOURCE_MODELS),5)
        self.assertEqual(n.BOOTSTRAP_DRAWS,4096)


if __name__=="__main__":
    unittest.main()
