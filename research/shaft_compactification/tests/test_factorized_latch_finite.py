import unittest
from research.shaft_compactification.factorized_latch_finite import (
    reduced, reduced_step, canonical_two, reduced_two, reference
)


class FactorizedLatchFiniteTests(unittest.TestCase):
    def test_gauge_invariance(self):
        for g in (0,1,33,63):
            for l in (0,1):
                self.assertEqual(reduced((g,l,3)), reduced((g^63,l^1,3)))
    def test_two_query_reconstruction(self):
        for family in ('repeat_toggle','successor_toggle'):
            self.assertEqual(canonical_two(3,0,0,family),reduced_two(3,0,family))
    def test_causal_factor_update(self):
        for family in ('repeat_toggle','successor_toggle'):
            for prev in range(6):
                a=reduced_step((22,prev),2,family)
                self.assertEqual(len(a[0]),2)
                self.assertIn(a[1],(0,1))
    def test_exact_384_classes(self):
        r=reference()
        for f in r['result'].values():
            self.assertEqual(f['minimal_behavioral_classes'],384)
            self.assertEqual(f['two_query_signature_classes'],384)
            self.assertEqual(f['canonical_states_per_reduced_code'],2)
    def test_hidden_p_distinguishable(self):
        for family in ('repeat_toggle','successor_toggle'):
            self.assertNotEqual(reduced_two(3,0,family),reduced_two(3,1,family))

if __name__=='__main__':unittest.main()
