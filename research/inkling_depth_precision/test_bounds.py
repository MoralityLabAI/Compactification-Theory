"""Exact source-only checks for recurrence bounds, not neural evidence."""

from __future__ import annotations

import unittest

try:
    from research.inkling_depth_precision.bounds import (
        applied_layer_proxy, error_envelope, first_certified_depth, fixed_gate_gain,
    )
except ModuleNotFoundError:
    from bounds import applied_layer_proxy, error_envelope, first_certified_depth, fixed_gate_gain


class TestDepthPrecision(unittest.TestCase):
    def test_contractive_rounding_envelope(self) -> None:
        self.assertEqual(error_envelope([0.5] * 3, [0.1] * 3),
                         (0.0, 0.1, 0.15000000000000002, 0.17500000000000002))
        self.assertLess(error_envelope([0.5] * 10, [0.1] * 10)[-1], 0.2)

    def test_expansive_error_and_distinct_gates(self) -> None:
        self.assertEqual(error_envelope([2, 2], [0.1, 0.1]), (0.0, 0.1, 0.30000000000000004))
        self.assertEqual(fixed_gate_gain([0.25, -0.5, 0.1], 2), 1)
        self.assertRaises(ValueError, error_envelope, [1, 2], [0.1])
        self.assertRaises(ValueError, fixed_gate_gain, [], 1)

    def test_layer_cost_and_state_threshold(self) -> None:
        self.assertEqual(
            applied_layer_proxy(fixed_layers=12, looped_layers=12,
                                loop_passes=3, total_generated_tokens=100), 4800
        )
        self.assertEqual(first_certified_depth([0.5, 0.2, 0.1], 0.2), 1)
        self.assertIsNone(first_certified_depth([0.5, 0.2], 0.1))
        self.assertRaises(ValueError, applied_layer_proxy,
                          fixed_layers=12, looped_layers=12,
                          loop_passes=0, total_generated_tokens=10)


if __name__ == "__main__":
    unittest.main()
