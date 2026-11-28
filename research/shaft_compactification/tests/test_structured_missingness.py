import unittest

from research.shaft_compactification.structured_missingness import (
    censored_query_candidates, reference, systematic_relabeling_matches_wrong_gauge,
)
from research.shaft_compactification.noisy_gauge import BUDGETS


class StructuredMissingnessTests(unittest.TestCase):
    def test_two_hidden_query_coordinates_leave_permanent_gauge_ambiguity(self):
        for design in ("onehot", "balanced"):
            for budget in BUDGETS:
                self.assertEqual(len(censored_query_candidates(design, budget)), 2)

    def test_fully_observed_anchors_identify_unique_gauge(self):
        for design in ("onehot", "balanced"):
            self.assertEqual(len(censored_query_candidates(design, 3, (0,1,2,3))), 1)

    def test_structured_wrong_gauge_looks_perfectly_consistent(self):
        for design in ("onehot", "balanced"):
            for budget in BUDGETS:
                result = systematic_relabeling_matches_wrong_gauge(design, budget)
                self.assertTrue(result["different_from_true"])
                self.assertTrue(result["unique_wrong_gauge"])

    def test_invalid_visible_indices(self):
        with self.assertRaises(ValueError):
            censored_query_candidates("balanced", 3, (0, 4))

    def test_receipt_shape(self):
        receipt = reference()
        self.assertEqual(receipt["results"]["balanced"]["24"]
                         ["compatible_gauges_with_queries_2_3_never_seen"], 2)


if __name__ == "__main__":
    unittest.main()
