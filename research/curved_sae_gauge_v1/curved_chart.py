"""SAU-CURVE01: oracle curved chart, rank-one linear null and gauge transfer.

Independent analytic synthetic CPU witness. Not a trained SAE, nor an
identifiability theorem for multiple superposed manifolds.
Run: python curved_chart.py [--self-test]
"""
from __future__ import annotations

import json
import math
import random
import sys
import unittest

N = 64
TAU = 2.0 * math.pi
TRUE_FRAME = 1.31
STEP = 0.04


def rotate(v: tuple[float, float], angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def source(theta: float) -> tuple[float, float]:
    return rotate((math.cos(theta), math.sin(theta)), 0.37)


def target(theta: float) -> tuple[float, float]:
    return rotate(source(theta), TRUE_FRAME)


def calibrate(xs: list[tuple[float, float]],
              ys: list[tuple[float, float]]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError('paired calibration inputs required')
    dot = sum(x[0] * y[0] + x[1] * y[1] for x, y in zip(xs, ys))
    cross = sum(x[0] * y[1] - x[1] * y[0] for x, y in zip(xs, ys))
    if math.hypot(dot, cross) < 1e-10:
        raise ValueError('rotation undefined from calibration')
    return math.atan2(cross, dot)


def square_error(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def native_utility(x: tuple[float, float], frame: float) -> float:
    # Fixed native readout rotated with its physical residual-stream frame.
    w = rotate((0.8, 0.6), frame)
    return w[0] * x[0] + w[1] * x[1]


def tangent(theta: float) -> tuple[float, float]:
    return rotate((-math.sin(theta), math.cos(theta)), 0.37)


def audit() -> dict:
    # Calibration and test phases are disjoint. Assignments use calibration only.
    cal = [TAU * j / N for j in range(N)]
    test = [TAU * (j + 0.43) / N for j in range(N)]
    xs, ys = [source(t) for t in cal], [target(t) for t in cal]
    fitted = calibrate(xs, ys)
    shuffled = ys.copy()
    random.Random(20261117).shuffle(shuffled)
    shuffled_fitted = calibrate(xs, shuffled)
    fit_errors, null_errors, effects, null_effects = [], [], [], []
    line_errors, oracle_chart_errors = [], []
    for theta in test:
        a, b = source(theta), target(theta)
        fit_errors.append(square_error(rotate(a, fitted), b))
        null_errors.append(square_error(rotate(a, shuffled_fitted), b))
        # One fixed 1D linear subspace, compared with the oracle 1D circle chart.
        line_errors.append(a[1] ** 2)
        oracle_chart_errors.append(square_error(source(theta), a))
        direction = tangent(theta)
        true_delta = native_utility(tuple(b[k] + STEP * rotate(direction, TRUE_FRAME)[k]
                                          for k in range(2)), 0.37 + TRUE_FRAME) - \
                     native_utility(tuple(b[k] - STEP * rotate(direction, TRUE_FRAME)[k]
                                          for k in range(2)), 0.37 + TRUE_FRAME)
        fitted_delta = native_utility(tuple(b[k] + STEP * rotate(direction, fitted)[k]
                                            for k in range(2)), 0.37 + TRUE_FRAME) - \
                       native_utility(tuple(b[k] - STEP * rotate(direction, fitted)[k]
                                            for k in range(2)), 0.37 + TRUE_FRAME)
        null_delta = native_utility(tuple(b[k] + STEP * rotate(direction, shuffled_fitted)[k]
                                          for k in range(2)), 0.37 + TRUE_FRAME) - \
                     native_utility(tuple(b[k] - STEP * rotate(direction, shuffled_fitted)[k]
                                          for k in range(2)), 0.37 + TRUE_FRAME)
        effects.append(abs(fitted_delta - true_delta))
        null_effects.append(abs(null_delta - true_delta))
    mean = lambda a: sum(a) / len(a)
    return {'protocol': 'SAU-CURVE01', 'kind': 'synthetic_oracle_chart',
            'calibration_pairs': N, 'heldout_pairs': N,
            'rank_one_linear_residual_energy': mean(line_errors),
            'oracle_chart_residual_energy': mean(oracle_chart_errors),
            'matched_gauge_heldout_mse': mean(fit_errors),
            'shuffled_gauge_heldout_mse': mean(null_errors),
            'matched_native_tangent_mae': mean(effects),
            'shuffled_native_tangent_mae': mean(null_effects)}


class Tests(unittest.TestCase):
    def test_oracle_chart_rank_one_boundary(self):
        d = audit()
        self.assertAlmostEqual(d['rank_one_linear_residual_energy'], 0.5, places=12)
        self.assertEqual(d['oracle_chart_residual_energy'], 0.0)

    def test_calibration_is_source_only(self):
        d = audit()
        self.assertLess(d['matched_gauge_heldout_mse'], 1e-25)
        self.assertGreater(d['shuffled_gauge_heldout_mse'], 0.01)

    def test_native_intervention_control(self):
        d = audit()
        self.assertLess(d['matched_native_tangent_mae'], 1e-12)
        self.assertGreater(d['shuffled_native_tangent_mae'], 0.001)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        print(json.dumps(audit(), indent=2, sort_keys=True))
