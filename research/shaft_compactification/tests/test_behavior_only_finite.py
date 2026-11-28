import unittest
from itertools import product
from research.shaft_compactification.behavior_only_finite import decode_two,reference
from research.shaft_compactification.factorized_latch_finite import canonical_two,reduced

class BehaviorOnlyFiniteTests(unittest.TestCase):
    def test_exhaustive(self):
        self.assertTrue(all(x["all_behavioral_codes_exactly_recovered"] for x in reference()["result"].values()))
    def test_previous_inverse(self):
        for family in ("repeat_toggle","successor_toggle"):
            for p in range(6):
                self.assertEqual(decode_two(canonical_two(13,1,p,family),family),(13^63,p))
    def test_gauge_symmetry(self):
        for family in ("repeat_toggle","successor_toggle"):
            for p in range(6):
                self.assertEqual(decode_two(canonical_two(3,0,p,family),family),
                                 decode_two(canonical_two(60,1,p,family),family))
    def test_invalid_signature(self):
        with self.assertRaises(ValueError): decode_two((0,)*35,"repeat_toggle")
        with self.assertRaises(ValueError): decode_two((0,)*36,"repeat_toggle")
        with self.assertRaises(ValueError): decode_two((0,)*36,"other")
    def test_exact_all_states(self):
        for family in ("repeat_toggle","successor_toggle"):
            for g,l,p in product(range(64),(0,1),range(6)):
                self.assertEqual(decode_two(canonical_two(g,l,p,family),family),
                                 reduced((g,l,p)))

if __name__=="__main__":
    unittest.main()
