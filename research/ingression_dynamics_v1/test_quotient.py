import unittest
from fractions import Fraction
from research.ingression_dynamics_v1.quotient import (
    commutes, projected_path, macro_path, one_action_error, expected_return,
)

class IngressionQuotientTests(unittest.TestCase):
    def test_exact_projection_with_disturbance(self):
        projection = {0: 0, 1: 1, 2: 0, 3: 1}
        micro = {"flip": {0: 1, 1: 0, 2: 3, 3: 2},
                 "disturb": {0: 2, 1: 3, 2: 0, 3: 1}}
        macro = {"flip": {0: 1, 1: 0}, "disturb": {0: 0, 1: 1}}
        self.assertTrue(commutes(micro, macro, projection))
        for start in projection:
            for actions in (["flip", "flip"], ["disturb", "flip", "disturb", "flip"]):
                self.assertEqual(projected_path(start, actions, micro, projection),
                                 macro_path(projection[start], actions, macro))

    def test_fibre_obstruction(self):
        projection = {0: 0, 1: 0, 2: 1}
        micro = {"step": {0: 2, 1: 1, 2: 2}}
        for a in (0, 1):
            for b in (0, 1):
                self.assertFalse(commutes(micro, {"step": {0: a, 1: b}}, projection))

    def test_exact_rational_coupling_witness(self):
        q = Fraction
        projection = {0: 0, 1: 1}
        micro = {0: {0: q(3, 4), 1: q(1, 4)}, 1: {1: q(1)}}
        macro = {0: {0: q(1)}, 1: {1: q(1)}}
        error = one_action_error(micro, macro, projection)
        self.assertEqual(error, q(1, 4))
        actual = expected_return(micro, 0, {0: q(0), 1: q(1)}, 3)
        control = expected_return(macro, 0, {0: q(0), 1: q(1)}, 3)
        self.assertEqual(actual, q(11, 16))
        self.assertEqual(control, q(0))
        self.assertLessEqual(abs(actual-control), error*3*2/2)

if __name__ == "__main__":
    unittest.main()
