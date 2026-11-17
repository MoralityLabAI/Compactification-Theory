import unittest

from research.shaft_compactification.gauge_identifiability import (
    ANCHORS, GAUGES, candidates_for_goals, infer_gauges, reference, signature,
)


class GaugeIdentifiabilityTests(unittest.TestCase):
    def test_exact_anchor_counts(self):
        self.assertEqual([len(candidates_for_goals(ANCHORS[:k])) for k in range(4)],
                         [48, 6, 2, 1])

    def test_three_anchor_unique(self):
        self.assertEqual(candidates_for_goals(ANCHORS), [((1, 0, 3, 2), 1)])

    def test_two_anchors_never_identify_full_gauge(self):
        for first in range(16):
            for second in range(16):
                self.assertGreaterEqual(len(candidates_for_goals((first, second))), 2)

    def test_unpaired_goals_symmetric_under_all_gauges(self):
        original = {signature(goal) for goal in range(16)}
        self.assertEqual(len(original), 16)
        for perm, flip in GAUGES:
            self.assertEqual({signature(goal, perm, flip) for goal in range(16)}, original)

    def test_unequal_paired_observations_rejected(self):
        with self.assertRaises(ValueError):
            infer_gauges([(1, 0, 0, 0)], [])

    def test_reference_receipt(self):
        r = reference()
        self.assertEqual(r['candidate_count_by_anchor_number'], {'0':48,'1':6,'2':2,'3':1})
        self.assertEqual(r['two_anchor_pair_count_examined'], 256)


if __name__ == '__main__':
    unittest.main()
