"""CPU algebra/rounding tests. No model downloads or behavioral claims."""
import unittest
import numpy as np
from surgery import (plan_surgery, audit, orth, bf16_round, assert_disjoint_ids)


class SurgeryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261104)
        self.w = self.rng.normal(size=(12, 17))
        self.d = self.rng.normal(size=(12, 4))
        self.s = self.rng.normal(size=(12, 3))
        self.h = self.rng.normal(size=(17, 5))

    def test_one_sided_formula(self):
        p = plan_surgery(self.w, self.d, self.s)
        np.testing.assert_allclose(p.delta(), -p.q @ p.q.T @ self.w, atol=1e-12)
        self.assertTrue(audit(self.w, p.merge(self.w), p)["within_tolerance"])

    def test_two_sided_constraints(self):
        p = plan_surgery(self.w, self.d, self.s, self.h)
        delta = p.delta()
        np.testing.assert_allclose(delta @ p.h, 0, atol=1e-12)
        np.testing.assert_allclose(p.s.T @ delta, 0, atol=1e-12)
        residual = p.q.T @ (self.w + delta)
        np.testing.assert_allclose(residual, (p.q.T @ self.w @ p.h) @ p.h.T, atol=1e-12)

    def test_minimum_norm_pythagoras(self):
        p = plan_surgery(self.w, self.d, self.s, self.h)
        z = self.rng.normal(size=self.w.shape)
        z -= p.s @ (p.s.T @ z)
        z -= p.q @ (p.q.T @ z)
        z -= (z @ p.h) @ p.h.T
        self.assertAlmostEqual(np.vdot(p.delta(), z), 0, places=11)
        self.assertAlmostEqual(np.linalg.norm(p.delta()+z)**2,
                               np.linalg.norm(p.delta())**2+np.linalg.norm(z)**2, places=10)

    def test_independent_constraint_solver(self):
        w, d, s, h = self.w[:4, :5], self.d[:4, :1], self.s[:4, :1], self.h[:5, :1]
        p = plan_surgery(w, d, s, h)
        def constraint(x):
            r = p.q.T @ x
            return np.concatenate([(p.s.T @ x).ravel(), (x @ p.h).ravel(),
                                   (r - (r @ p.h) @ p.h.T).ravel()])
        operator = np.stack([constraint(e.reshape(w.shape)) for e in np.eye(w.size)], axis=1)
        target = constraint(p.delta())
        independent = np.linalg.lstsq(operator, target, rcond=1e-10)[0].reshape(w.shape)
        np.testing.assert_allclose(independent, p.delta(), atol=1e-12)

    def test_partial_strength(self):
        full = plan_surgery(self.w, self.d, self.s, self.h)
        half = plan_surgery(self.w, self.d, self.s, self.h, strength=0.5)
        np.testing.assert_allclose(half.delta(), 0.5*full.delta(), atol=1e-12)
        self.assertTrue(audit(self.w, half.merge(self.w), half)["within_tolerance"])

    def test_zero_strength(self):
        p = plan_surgery(self.w, self.d, strength=0)
        np.testing.assert_array_equal(p.merge(self.w), self.w)

    def test_rank_cap_and_factor_equality(self):
        p = plan_surgery(self.w, self.d, self.s, self.h, rank=2)
        self.assertLessEqual(np.linalg.matrix_rank(p.delta(), tol=1e-10), 2)
        x = self.rng.normal(size=(17, 8))
        np.testing.assert_allclose(p.merge(self.w) @ x, self.w @ x + p.b @ (p.a @ x), atol=1e-12)

    def test_all_failure_protected(self):
        p = plan_surgery(self.w, self.s, self.s)
        self.assertEqual(p.q.shape[1], 0)
        np.testing.assert_array_equal(p.merge(self.w), self.w)

    def test_full_input_protection(self):
        p = plan_surgery(self.w, self.d, h=np.eye(17))
        np.testing.assert_allclose(p.delta(), 0, atol=1e-12)

    def test_no_directions(self):
        p = plan_surgery(self.w, np.empty((12, 0)))
        self.assertEqual(p.q.shape[1], 0)
        np.testing.assert_array_equal(p.merge(self.w), self.w)

    def test_near_overlap_rejected_at_original_scale(self):
        s = np.eye(12)[:, :1]
        d = s + 1e-14 * np.eye(12)[:, 1:2]
        self.assertEqual(plan_surgery(self.w, d, s).q.shape[1], 0)

    def test_near_overlap_retained_orthogonally(self):
        s = orth(self.s[:, :1])
        v = self.rng.normal(size=(12, 1)); v -= s @ (s.T @ v)
        p = plan_surgery(self.w, s+1e-7*v, s)
        self.assertEqual(p.q.shape[1], 1)
        np.testing.assert_allclose(p.s.T @ p.q, 0, atol=1e-12)

    def test_rank_deficient_protection(self):
        p = plan_surgery(self.w, self.d, np.hstack([self.s, self.s]))
        self.assertEqual(p.s.shape[1], 3)
        self.assertTrue(audit(self.w, p.merge(self.w), p)["within_tolerance"])

    def test_quantization_audit(self):
        w = bf16_round(self.w)
        p = plan_surgery(w, self.d, self.s, self.h)
        exact, cast = audit(w, p.merge(w), p), audit(w, bf16_round(p.merge(w)), p)
        self.assertTrue(exact["within_tolerance"])
        self.assertFalse(cast["within_tolerance"])
        self.assertGreater(cast["merge_deviation_fro"], 0)

    def test_bf16_ties_and_idempotence(self):
        x = np.array([[1+1/256, 1+3/256, -1-1/256, 0.0]])
        expected = np.array([[1.0, 1+2/128, -1.0, 0.0]])
        np.testing.assert_array_equal(bf16_round(x), expected)
        y = bf16_round(self.w)
        np.testing.assert_array_equal(bf16_round(y), y)

    def test_invalid_inputs(self):
        for strength in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError): plan_surgery(self.w, self.d, strength=strength)
        for rank in [-1, 0.5, True]:
            with self.assertRaises(ValueError): plan_surgery(self.w, self.d, rank=rank)
        for d in [np.ones((11, 1)), np.full((12, 1), np.nan), np.ones((12, 1))*1j]:
            with self.assertRaises(ValueError): plan_surgery(self.w, d)
        with self.assertRaises(ValueError): plan_surgery(self.w, self.d, h=np.ones((12, 1)))
        with self.assertRaises(ValueError): orth(self.d, rtol=-1)
        with self.assertRaises(ValueError): bf16_round(np.array([[1e100]]))

    def test_source_binding_and_immutability(self):
        p = plan_surgery(self.w, self.d)
        with self.assertRaises(ValueError): p.merge(self.w+1)
        with self.assertRaises(ValueError): p.a[0, 0] = 0
        with self.assertRaises(ValueError): audit(self.w, np.ones((2, 2)), p)

    def test_split_leakage_guard(self):
        assert_disjoint_ids(design=["a", "b"], validation=["c"], test=["d"])
        for splits in [dict(design=["a"], test=["a"]), dict(test=["a", "a"]), dict(test=[""])]:
            with self.assertRaises(ValueError): assert_disjoint_ids(**splits)

    def test_sink_counterexample(self):
        w = np.eye(3); s = w[:, :1]
        d = np.array([[-0.77], [np.sqrt(1-0.77**2)], [0]])
        naive = plan_surgery(w, d); protected = plan_surgery(w, d, s)
        self.assertAlmostEqual(np.linalg.norm(s.T @ naive.delta()), 0.77)
        self.assertLess(np.linalg.norm(s.T @ protected.delta()), 1e-12)
        self.assertAlmostEqual(protected.retained_failure_energy, 1-0.77**2)

    def test_rank_zero(self):
        p = plan_surgery(self.w, self.d, rank=0)
        np.testing.assert_array_equal(p.merge(self.w), self.w)

    def test_rounding_error_bounds(self):
        w = bf16_round(self.w)
        p = plan_surgery(w, self.d, self.s, self.h)
        r = audit(w, bf16_round(p.merge(w)), p)
        for key in ("protected_output_error_fro", "protected_input_error_fro",
                    "target_constraint_error_fro"):
            self.assertLessEqual(r[key], r["merge_deviation_fro"] + 1e-12)

    def test_bias_and_branch_hook(self):
        p = plan_surgery(self.w, self.d, self.s, self.h)
        x, bias = self.rng.normal(size=(17, 7)), self.rng.normal(size=(12, 1))
        merged = p.merge(self.w) @ x + bias
        hooked = self.w @ x + bias + p.b @ (p.a @ x)
        np.testing.assert_allclose(merged, hooked, atol=1e-12)

    def test_global_ablation_conflict(self):
        w = np.eye(3)
        p = plan_surgery(w, w[:, :1], h=w[:, :1])
        np.testing.assert_allclose(p.delta(), 0, atol=1e-12)
        r = audit(w, p.merge(w), p)
        self.assertAlmostEqual(r["remaining_q_output_fro"], 1)
        self.assertAlmostEqual(r["protected_input_residual_floor_fro"], 1)

    def test_multiseed_property_sweep(self):
        for seed in range(25):
            rng = np.random.default_rng(seed)
            w, d = rng.normal(size=(13, 19)), rng.normal(size=(13, 5))
            p = plan_surgery(w, d, rng.normal(size=(13, 3)), rng.normal(size=(19, 4)), rank=3)
            self.assertTrue(audit(w, p.merge(w), p)["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
