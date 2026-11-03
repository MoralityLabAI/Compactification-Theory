"""Finite smoke tests for the halting-to-geometry Lean construction.

An activated stream map reads input coordinate 2*n at output coordinate n;
before activation, it reads coordinate n. A finite truncation is a computable
map on finite marked spaces. These tests check numerical witnesses, never
substitute for mathlib's theorem about partial-recursive programs.
"""
from itertools import product
import unittest
from filtered import MarkedSpace, Interpretation


def accelerated_map(target_width: int, activates_at: int | None) -> Interpretation:
    if target_width < 1:
        raise ValueError("Target width must be positive.")
    source_width = 2 * target_width - 1
    source = MarkedSpace(tuple(product((0, 1), repeat=source_width)))
    target = MarkedSpace(tuple(product((0, 1), repeat=target_width)))
    indices = tuple(n if activates_at is None or n < activates_at else 2 * n
                    for n in range(target_width))
    return Interpretation(source, target,
                          tuple(target.points.index(tuple(bits[k] for k in indices))
                                for bits in source.points))


class HaltingGeometryFiniteTests(unittest.TestCase):
    def test_unactivated_identity_prefix_has_constant_one(self):
        for width in (1, 2, 3, 4):
            interp = accelerated_map(width, None)
            self.assertEqual(interp.lipschitz_constant(), 1 if width else 0)
            self.assertEqual(interp.least_delay(), 0)
            self.assertEqual(interp.profile_constant(), 1)

    def test_eventual_activation_distortion_grows(self):
        for width in (3, 4):
            interp = accelerated_map(width, 2)
            expected = 2 ** (width - 1)
            self.assertEqual(interp.lipschitz_constant(), expected)
            self.assertEqual(interp.profile_constant(), expected)
            self.assertEqual(interp.least_delay(), width - 1)
            _, attained = interp.optimal_pullback_witness()
            self.assertEqual(attained, expected)

    def test_stage_sensitivity(self):
        width = 4
        early = accelerated_map(width, 1)
        late = accelerated_map(width, 3)
        never = accelerated_map(width, None)
        self.assertEqual(early.lipschitz_constant(), 8)
        self.assertEqual(late.lipschitz_constant(), 8)
        self.assertEqual(never.lipschitz_constant(), 1)
        self.assertEqual(early.depth_profile(), (0, 1, 3, 5, 7))
        self.assertEqual(late.depth_profile(), (0, 1, 2, 3, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
