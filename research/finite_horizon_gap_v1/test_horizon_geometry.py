"""FHC-01B exact + empirical regression tests."""
from __future__ import annotations

import math
import unittest

import numpy as np

from horizon_geometry import (
    HORIZONS, PERIOD, WIDTH, action_refinement, full_certificate,
    observation_masks, primitive_shift6, symbolic_quotient,
    trace_from_state, walsh_orthogonality,
)
from bench_future_metric import (
    SEED, calibration_scale, distortion_metrics, exact_binary_code,
    features, run, run_one, split_indices,
)
from research.recurrent_circuits_v1.reference import row_times_matrix


class ExactFutureGeometry(unittest.TestCase):
    def test_primitive_period_exhausts_every_nonzero_mask(self):
        masks = observation_masks(63)
        self.assertEqual(len(masks), 63)
        self.assertEqual(set(masks), set(range(1, 64)))
        self.assertEqual(row_times_matrix(masks[-1], primitive_shift6()), 1)

    def test_exact_minimum_binary_bits_and_trace_counts(self):
        for t in HORIZONS:
            q = symbolic_quotient(t)
            self.assertEqual(q["rank_gf2"], min(WIDTH, t))
            self.assertEqual(q["distinct_traces"], 2 ** min(WIDTH, t))

    def test_independent_walsh_features_have_exact_real_rank(self):
        for t in HORIZONS:
            witness = walsh_orthogonality(t)
            self.assertEqual(witness["real_centered_feature_rank"], t)
            self.assertEqual(witness["distinct_masks"], t)
            self.assertEqual(witness["centered_covariance_diagonal"], "1/4")

    def test_exact_regular_simplex_at_63(self):
        x = features(63)
        self.assertEqual(x.shape, (64, 63))
        for i in range(64):
            for j in range(i+1, 64):
                self.assertEqual(float(np.sum((x[i]-x[j])**2)), 32.0)
        matrix = np.cov(x, rowvar=False, bias=True)
        self.assertTrue(np.allclose(matrix, np.eye(63) / 4, atol=1e-12))

    def test_intervention_interface_exposes_all_coordinates(self):
        receipt = action_refinement()
        self.assertEqual(receipt["plain_observation_bits"], 1)
        self.assertEqual(receipt["all_rotations_observation_bits"], 6)
        self.assertTrue(receipt["plain_pair_equal"])
        self.assertTrue(receipt["post_rotation_pair_differs"])

    def test_binary_code_and_every_future_trace_decodable(self):
        for t in (1, 4, 6, 16, 63):
            c = exact_binary_code(t)
            self.assertEqual(c.shape, (64, min(t, 6)))
            self.assertEqual(len(set(map(tuple, c))), 2**min(t, 6))
            masks = symbolic_quotient(t)["decoder_masks"]
            for x in range(64):
                state = sum(int(bit) << i for i, bit in enumerate(c[x]))
                prediction = tuple((state & row).bit_count()%2 for row in masks)
                self.assertEqual(prediction,
                                 trace_from_state(x, observation_masks(t)))

    def test_rejects_invalid_horizons(self):
        for T in (-1, 64, 1.5, True):
            with self.assertRaises(ValueError):
                observation_masks(T)

    def test_receipt_provenance(self):
        record = full_certificate()
        self.assertEqual(record["rc01_dependency"],
                         "research.recurrent_circuits_v1.reference")
        self.assertEqual(record["full_state_count"], 64)
        self.assertFalse(record["source_family_098_witness_constructed"])


class MeasuredFiniteMetric(unittest.TestCase):
    def test_holdout_state_split_is_deterministic_and_disjoint(self):
        a, b = split_indices(SEED)
        c, d = split_indices(SEED)
        self.assertTrue(np.array_equal(a, c))
        self.assertTrue(np.array_equal(b, d))
        self.assertEqual((len(a), len(b)), (48, 16))
        self.assertEqual(set(a).union(set(b)), set(range(64)))
        self.assertFalse(set(a).intersection(set(b)))

    def test_calibration_only_scale_free_of_test_data(self):
        train, test = split_indices()
        x = features(16)
        c = exact_binary_code(16)
        scale = calibration_scale(x[train], c[train])
        _ = distortion_metrics(x[test], c[test], scale, test)
        self.assertEqual(scale, calibration_scale(x[train], c[train]))

    def test_identity_embedding_never_has_violations(self):
        train, test = split_indices()
        for T in HORIZONS:
            x = features(T)
            scale = calibration_scale(x[train], x[train])
            m = distortion_metrics(x[test], x[test], scale, test)
            self.assertAlmostEqual(scale, 1.0)
            self.assertEqual(m["violation_counts_calibration_scale"],
                             {"1.5": 0, "2.0": 0})
            self.assertEqual(m["optimal_rescaling_distortion_heldout"], 1.0)

    def test_some_exact_semantic_code_distorts_future_metric(self):
        train, test = split_indices()
        full = features(63)
        code = exact_binary_code(63)
        scale = calibration_scale(full[train], code[train])
        m = distortion_metrics(full[test], code[test], scale, test)
        self.assertGreater(m["optimal_rescaling_distortion_heldout"], 2)
        self.assertLessEqual(m["optimal_rescaling_distortion_heldout"],
                             math.sqrt(6) + 1e-10)

    def test_full_run_contains_pca_gaussian_and_exact_code(self):
        result = run(horizons=(1, 4, 16, 63), widths=(2, 4, 8, 16))
        self.assertEqual(result["status"],
                         "FROZEN_SYNTHETIC_RECURRENT_PILOT")
        for row in result["records"]:
            names = {x["method"] for x in row["methods"]}
            self.assertIn("exact_binary_quotient", names)
            self.assertIn("full_trace_identity", names)
            if row["horizon"] >= 16:
                self.assertIn("pca_8", names)
                self.assertIn("gaussian_8", names)
            for method in row["methods"]:
                metric = method["heldout"]
                self.assertEqual(
                    metric["pairs_distinct"],
                    sum(1 for i in range(16) for j in range(i+1, 16)
                        if np.linalg.norm(features(row["horizon"])[
                            result["heldout_state_ids"][i]] -
                            features(row["horizon"])[
                                result["heldout_state_ids"][j]]) > 0))
        self.assertFalse(result["checkpoint_evaluated"])

    def test_heldout_simplex_is_invisible_to_training_pca(self):
        # Exact simplex geometry: training-centered differences are
        # perpendicular to differences among any entirely unseen vertices.
        train, test = split_indices()
        future = features(63)
        centered_train = future[train] - future[train].mean(axis=0)
        for i in range(len(test)):
            for j in range(i+1, len(test)):
                unseen_difference = future[test[i]] - future[test[j]]
                self.assertTrue(np.allclose(
                    centered_train @ unseen_difference, 0, atol=1e-12))
        _, s, vt = np.linalg.svd(centered_train, full_matrices=False)
        self.assertEqual(int(np.sum(s > 1e-10)), len(train)-1)
        # Every positive-variance PCA component annihilates every
        # difference between two held-out vertices.
        for d in (2, 4, 6, 8, 16, 32):
            projected = future[test] @ vt[:d].T
            self.assertLess(np.max(np.abs(projected - projected[:1])), 1e-12)

    def test_failed_pair_is_reproducible(self):
        train, test = split_indices()
        full, code = features(63), exact_binary_code(63)
        scale = calibration_scale(full[train], code[train])
        m = distortion_metrics(full[test], code[test], scale, test)
        for D, pair in m["violation_witnesses"].items():
            i, j = pair["state_a"], pair["state_b"]
            da = float(np.linalg.norm(full[i]-full[j]))
            db = float(np.linalg.norm(code[i]-code[j]))
            self.assertAlmostEqual(pair["source_distance"], da)
            self.assertAlmostEqual(pair["code_distance"], db)
            self.assertAlmostEqual(pair["ratio_scaled"], scale*db/da)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            run(horizons=(64,))
        with self.assertRaises(ValueError):
            distortion_metrics(np.zeros((3,2)), np.zeros((2,1)),
                               1, np.arange(3))


if __name__ == "__main__":
    unittest.main()
