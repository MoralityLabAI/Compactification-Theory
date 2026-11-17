import unittest

from research.shaft_compactification.noisy_gauge import (
    BALANCED, BUDGETS, GAUGES, TRUE, ONEHOT, anchors_for, chernoff_union_bound,
    condition, distance_profile, minimax_three_anchors, packed_response, reference,
    signature_bits,
)


class NoisyGaugeTests(unittest.TestCase):
    def test_gauge_count(self):
        self.assertEqual(len(GAUGES), 48)
        self.assertEqual(len(set(GAUGES)), 48)

    def test_three_anchor_minimax(self):
        r = minimax_three_anchors()
        self.assertEqual(r["max_min_distance"], 4)
        self.assertEqual(r["distinct_triplets_checked"], 560)
        self.assertEqual(len(r["optimal_triples"]), 8)
        self.assertIn([3, 5, 6], r["optimal_triples"])

    def test_balanced_doubles_minimum_separation(self):
        self.assertEqual([min(distance_profile(anchors_for("onehot", b))) for b in BUDGETS],
                         [2, 4, 8, 16])
        self.assertEqual([min(distance_profile(anchors_for("balanced", b))) for b in BUDGETS],
                         [4, 8, 16, 32])

    def test_clean_complete_recovery(self):
        for design in ("onehot", "balanced"):
            result = condition(design, 0, 0, 32)
            for b in BUDGETS:
                self.assertEqual(result[str(b)]["unique_correct_recoveries"], 32)
                self.assertEqual(result[str(b)]["sufficient_margin_certificates"], 32)
                self.assertEqual(result[str(b)]["mean_posterior_gauge_entropy_bits"], 0)
                self.assertEqual(result[str(b)]["chernoff_union_bound_probability"], 0)

    def test_missing_completely_all_gauges_tied(self):
        distances = distance_profile(anchors_for("balanced", 3))
        self.assertEqual(chernoff_union_bound(distances, 0, 1), 1)

    def test_half_noise_is_invalid_for_claimed_bound(self):
        with self.assertRaises(ValueError):
            chernoff_union_bound([4, 6], 0.5, 0)

    def test_packed_code_consistency(self):
        anchors = (3, 5, 6)
        code = packed_response(anchors, TRUE)
        expected = [b for a in anchors for b in signature_bits(a, *TRUE)]
        self.assertEqual([(code >> j) & 1 for j in range(12)], expected)

    def test_randomized_condition_sanity(self):
        r = condition("balanced", 10, 25, 32)
        for b in BUDGETS:
            cell = r[str(b)]
            self.assertLessEqual(cell["unique_correct_recoveries"],
                                 cell["true_in_minimum_score_set"])
            self.assertLessEqual(cell["sufficient_margin_certificates"],
                                 cell["unique_correct_recoveries"])
            self.assertGreaterEqual(cell["mean_posterior_gauge_entropy_bits"], 0)
            self.assertTrue(0 <= cell["chernoff_union_bound_probability"] <= 1)

    def test_rejected_parameters(self):
        with self.assertRaises(ValueError):
            anchors_for("other", 3)
        with self.assertRaises(ValueError):
            condition("onehot", 11, 25, 16)

    def test_reference_shape(self):
        r = reference(8)
        self.assertEqual(r["trials_per_condition"], 8)
        self.assertEqual(len(r["designs"]["balanced"]["conditions"]), 12)
        self.assertEqual(r["gauge_count"], 48)


if __name__ == "__main__":
    unittest.main()
