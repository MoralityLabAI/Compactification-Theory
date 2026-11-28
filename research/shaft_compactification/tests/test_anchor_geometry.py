import unittest

from research.shaft_compactification.anchor_geometry import (
    select_minimax_triple, exact_target_bits, clean_gauge_margins,
)
from research.shaft_compactification.gauge_identifiability import (
    GAUGES, TARGET_PERM, TARGET_FLIP,
)


class AnchorGeometryTests(unittest.TestCase):
    def test_no_information_yields_zero_margin(self):
        r = select_minimax_triple([[0.0] * 48 for _ in range(16)], 10)
        self.assertEqual(r["minimum_expected_score_margin"], 0)
        self.assertEqual(r["selected_triple"], [0, 1, 2])

    def test_task_heterogeneity_changes_selected_design(self):
        scores = [[0.0] * 4 for _ in range(16)]
        for goal in range(16):
            for gauge in (1, 2, 3):
                scores[goal][gauge] = 3.0 if goal in (4,5,6) else -1.0
        r = select_minimax_triple(scores, 0)
        self.assertEqual(r["selected_triple"], [4,5,6])
        self.assertEqual(r["minimum_expected_score_margin"], 9)

    def test_undistinguishable_candidate_remains_uncertified(self):
        rows = [[0,1,0] for _ in range(16)]
        self.assertEqual(select_minimax_triple(rows, 0)["minimum_expected_score_margin"], 0)

    def test_matrix_shape(self):
        with self.assertRaises(ValueError):
            select_minimax_triple([[0,1]], 0)
        with self.assertRaises(ValueError):
            select_minimax_triple([[0,1]] * 16, 3)

    def test_clean_optimum_matches_exhaustive_gauge_geometry(self):
        truth = GAUGES.index((TARGET_PERM, TARGET_FLIP))
        margins = clean_gauge_margins(GAUGES, truth)
        optimum = select_minimax_triple(margins, truth)
        self.assertEqual(optimum["candidate_triples_considered"], 560)
        self.assertEqual(optimum["minimum_expected_score_margin"], 4)
        self.assertEqual(optimum["selected_triple"], [3, 5, 6])

    def test_query_mapping_orientation(self):
        self.assertEqual(exact_target_bits(1, TARGET_PERM, TARGET_FLIP),
                         (1,0,1,1))


if __name__ == "__main__":
    unittest.main()
