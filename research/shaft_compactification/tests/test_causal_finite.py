import unittest
from itertools import product
from research.shaft_compactification.causal_finite import (
    STATES, step, code, observe, reference,
)


class CausalFiniteTests(unittest.TestCase):
    def test_delayed_observation(self):
        left, right = (0,0,0), (0,0,1)
        self.assertEqual(code(left), code(right))
        self.assertEqual(observe(left,(1,)), observe(right,(1,)))
        self.assertNotEqual(observe(left,(1,1)), observe(right,(1,1)))

    def test_no_single_bit_decoder_can_restore_every_future(self):
        for x,y in product(STATES, repeat=2):
            decoder={0:x,1:y}
            self.assertFalse(all(
                observe(decoder[code(s)], (1,1))==observe(s,(1,1))
                for s in STATES))

    def test_full_state_code_is_causally_sufficient(self):
        for s in STATES:
            for n in range(5):
                for actions in product((0,1), repeat=n):
                    self.assertEqual(observe(s,actions), observe(tuple(s), actions))

    def test_reference(self):
        r=reference()
        self.assertEqual(r['universally_sound_visible_bit_decoders'],0)
        self.assertEqual(r['possible_visible_bit_decoders'],64)
        self.assertEqual(r['number_of_ordered_same_code_pairs_distinguishable_at_two_steps'],16)


if __name__ == '__main__':
    unittest.main()
