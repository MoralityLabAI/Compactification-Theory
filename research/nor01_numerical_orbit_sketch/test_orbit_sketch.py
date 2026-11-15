"""Local exact-gate and empirical-distortion regression tests for NOR-01."""
import math
from fractions import Fraction
import unittest

from orbit_sketch import BlockLoop, JordanBlock, norm, run_case


class OrbitSketchTests(unittest.TestCase):
    def setUp(self):
        self.block = JordanBlock(Fraction(3, 8), Fraction(1))
        self.model = BlockLoop((self.block,) * 4)

    def test_exact_rational_numerical_radius(self):
        self.assertEqual(self.model.q, Fraction(7, 8))
        self.assertEqual(self.model.eigenvalue_radius, Fraction(3, 8))
        self.assertGreater(self.model.operator_norm, 1.0)
        self.model.require_tail_certificate()

    def test_exact_power_matches_iteration(self):
        x0 = [z for _ in range(4) for z in (0.0, 0.5)]
        v = x0
        for t in range(16):
            closed = self.model.exact_power_action(x0, t)
            for u, w in zip(v, closed, strict=True):
                self.assertAlmostEqual(u, w, places=12)
            v = self.model.step(v)

    def test_crouzeix_special_case_power_gate(self):
        v = [z for _ in range(4) for z in (0.0, 0.5)]
        for t in range(35):
            observed = norm(self.model.exact_power_action(v, t))
            theoretical = 2 * float(self.model.q) ** t * norm(v)
            self.assertLessEqual(observed, theoretical + 1e-13)

    def test_spectral_only_gate_fails(self):
        v = [z for _ in range(4) for z in (0.0, 0.5)]
        self.assertGreater(norm(self.model.step(v)), norm(v))
        self.assertGreater(norm(self.model.step(v)),
                           2 * float(self.model.eigenvalue_radius) * norm(v))

    def test_reject_large_numerical_range_despite_small_spectrum(self):
        bad = BlockLoop((JordanBlock(Fraction(3, 8), Fraction(2)),))
        self.assertLess(bad.eigenvalue_radius, 1)
        self.assertGreater(bad.q, 1)
        with self.assertRaises(ValueError):
            bad.require_tail_certificate()

    def test_lp2_stitched_all_pairs(self):
        receipt = run_case(block_count=24, cutoff=20, k=24, p=2, seed=42,
                           tail_checks=16)
        self.assertTrue(receipt["finite_probe_within_theorem_bound"])
        self.assertTrue(receipt["spectral_shortcut_fails"])
        self.assertGreater(receipt["window_max_absolute_error"], 0)

    def test_lp4_stitched_all_pairs_and_compression(self):
        receipt = run_case(block_count=32, cutoff=18, k=24, p=4, seed=325094,
                           tail_checks=16)
        self.assertLess(receipt["sketch_dimension"], receipt["ambient_dimension"])
        self.assertTrue(receipt["finite_probe_within_theorem_bound"])
        self.assertTrue(math.isfinite(receipt["all_horizon_absolute_error_bound"]))


if __name__ == "__main__":
    unittest.main()
