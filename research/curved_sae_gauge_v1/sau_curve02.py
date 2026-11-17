"""SAU-CURVE02: tangent-frame identifiability under superposed curves.

Analytic two-curve R4 toy; no trained SAE, optimizer, or language-model weights.
Run: python research/curved_sae_gauge_v1/sau_curve02.py
"""
from __future__ import annotations

import json
import math
import unittest

import numpy as np

SEEDS = (103, 131, 167, 211)
N = 4096
ANGLES = (0, 5, 15, 30, 60, 90)
COVERAGE = (0.25, 0.50, 1.00)
THRESHOLD = 0.10


def tangent(theta: float, tilt_degrees: float, second: bool) -> np.ndarray:
    a = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    if second:
        c, s = math.cos(math.radians(tilt_degrees)), math.sin(math.radians(tilt_degrees))
        return np.concatenate([c * a, s * a])
    return np.concatenate([a, np.zeros(2)])


def sigma_min(theta1: np.ndarray, theta2: np.ndarray, tilt_degrees: float) -> np.ndarray:
    """Nonzero smaller singular value of the 4x2 active tangent frame."""
    dot = math.cos(math.radians(tilt_degrees)) * np.cos(theta1 - theta2)
    return np.sqrt(np.maximum(0.0, 1.0 - np.abs(dot)))


def result_for_seed(seed: int, tilt_degrees: float, coactive_fraction: float, n: int = N) -> dict:
    if not 0 <= coactive_fraction <= 1:
        raise ValueError("coactivation fraction must be in [0,1]")
    rng = np.random.default_rng(13 * seed + 3)
    a = rng.uniform(0, 2 * math.pi, n)
    b = rng.uniform(0, 2 * math.pi, n)
    active = rng.random(n) < coactive_fraction
    sv = np.ones(n)
    sv[active] = sigma_min(a[active], b[active], tilt_degrees)
    # Exact infimum under paired tangent phases, independent of sample phases.
    bound = math.sqrt(max(0.0, 1.0 - math.cos(math.radians(tilt_degrees))))
    return {"seed": seed, "tilt_deg": tilt_degrees, "coactivation": coactive_fraction,
            "n": n, "n_joint": int(active.sum()),
            "sample_min_sigma": round(float(sv.min()), 6),
            "q05_sigma": round(float(np.quantile(sv, 0.05)), 6),
            "median_sigma": round(float(np.median(sv)), 6),
            "fraction_sigma_below_010": round(float(np.mean(sv < THRESHOLD)), 6),
            "joint_infimum_sigma": round(bound, 6)}


def run() -> dict:
    rows = [result_for_seed(seed, tilt, cover) for tilt in ANGLES
            for cover in COVERAGE for seed in SEEDS]
    pooled = []
    for tilt in ANGLES:
        for cover in COVERAGE:
            relevant = [r for r in rows if r["tilt_deg"] == tilt and r["coactivation"] == cover]
            pooled.append({"tilt_deg": tilt, "coactivation": cover,
                           "n_total": sum(r["n"] for r in relevant),
                           "joint_n_total": sum(r["n_joint"] for r in relevant),
                           "mean_q05_sigma": round(float(np.mean([r["q05_sigma"] for r in relevant])), 6),
                           "mean_fraction_sigma_below_010": round(float(np.mean(
                               [r["fraction_sigma_below_010"] for r in relevant])), 6),
                           "joint_infimum_sigma": relevant[0]["joint_infimum_sigma"]})
    return {"protocol": "SAU-CURVE02", "type": "synthetic_analytic_two_atom",
            "seeds": SEEDS, "phase_pairs_per_seed": N, "angles_deg": ANGLES,
            "coactivation": COVERAGE, "sigma_threshold": THRESHOLD,
            "pooled": pooled, "seedwise": rows}


class TestTangentFrame(unittest.TestCase):
    def test_closed_form_matches_svd(self):
        rng = np.random.default_rng(991)
        for angle in ANGLES:
            for _ in range(20):
                a, b = rng.uniform(0, 2 * math.pi, 2)
                got = float(sigma_min(np.array([a]), np.array([b]), angle)[0])
                expected = np.linalg.svd(np.stack([tangent(a, angle, False),
                                                    tangent(b, angle, True)], axis=1),
                                         compute_uv=False)[-1]
                self.assertAlmostEqual(got, expected, places=10)

    def test_angle_limits_and_coactivation(self):
        a = np.array([0.0, 0.1, 1.0])
        b = a.copy()
        self.assertTrue(np.allclose(sigma_min(a, b, 0), 0))
        self.assertTrue(np.allclose(sigma_min(a, b, 90), 1))
        for seed in SEEDS:
            self.assertEqual(result_for_seed(seed, 0, 0, n=256)["sample_min_sigma"], 1.0)

    def test_monotone_infima_and_strong_angle_gate(self):
        bounds = [math.sqrt(max(0, 1 - math.cos(math.radians(a)))) for a in ANGLES]
        self.assertEqual(bounds[0], 0)
        self.assertTrue(all(bounds[j] < bounds[j + 1] for j in range(len(bounds) - 1)))
        for a in (15, 30, 60, 90):
            r = result_for_seed(103, a, 1.0)
            self.assertEqual(r["fraction_sigma_below_010"], 0)

    def test_coactivation_shares_risk(self):
        row = [result_for_seed(103, 0, c) for c in COVERAGE]
        self.assertLess(row[0]["fraction_sigma_below_010"], row[1]["fraction_sigma_below_010"])
        self.assertLess(row[1]["fraction_sigma_below_010"], row[2]["fraction_sigma_below_010"])


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
