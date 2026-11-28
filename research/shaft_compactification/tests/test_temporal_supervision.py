import unittest
from itertools import product
from research.shaft_compactification.temporal_supervision_finite import (
    supervised_trace, independent_trace, reference,
)


class TemporalSupervisionTests(unittest.TestCase):
    def test_repeat_output_is_pre_update(self):
        self.assertEqual(supervised_trace(0, (0, 0, 1), "repeat_toggle"),
                         ((0, 1, 0), (1, 0, 0), (0, 0, 1)))

    def test_successor_output_is_pre_update(self):
        self.assertEqual(supervised_trace(0, (1, 2, 0), "successor_toggle"),
                         ((0, 1, 0), (1, 0, 0), (1, 2, 0)))

    def test_all_short_traces(self):
        for family in ("repeat_toggle", "successor_toggle"):
            for goal in (0, 1, 7, 63):
                for n in range(4):
                    for tokens in product(range(6), repeat=n):
                        self.assertEqual(supervised_trace(goal, tokens, family),
                                         independent_trace(goal, tokens, family))

    def test_reference(self):
        r = reference()
        self.assertEqual(r["traces_checked"], 2 * 64 * (1 + 6 + 36 + 216))
        for x in r["separations"].values():
            self.assertEqual(x["one_query_aliases"], 3840)
            self.assertEqual(x["two_query_splits"], 3840)


if __name__ == "__main__":
    unittest.main()
