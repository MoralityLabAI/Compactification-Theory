import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monoshot as M


def rand_spd(rng, n):
    A = rng.normal(size=(n, n))
    return A @ A.T / n + 0.1 * np.eye(n)


class MonoShot(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_constraint_and_optimality(self):
        n, d, m = 9, 4, 2
        C, K, V = rand_spd(self.rng, n), self.rng.normal(size=(n, m)), self.rng.normal(size=(d, m))
        D = M.min_energy_edit(K, V, C)
        np.testing.assert_allclose(D @ K, V, atol=1e-10)
        self.assertLessEqual(np.linalg.matrix_rank(D), m)
        self.assertAlmostEqual(M.edit_cost(D, C), M.optimal_cost(K, V, C), places=10)
        for _ in range(50):
            Z = self.rng.normal(size=(d, n))
            Z -= (Z @ K) @ np.linalg.pinv(K)              # feasible direction: Z K = 0
            self.assertGreater(M.edit_cost(D + 0.3 * Z, C), M.edit_cost(D, C))

    def test_single_key_q_and_rank_one(self):
        n = 7
        C, k, v = rand_spd(self.rng, n), self.rng.normal(size=n), self.rng.normal(size=3)
        D = M.min_energy_edit(k[:, None], v[:, None], C)
        self.assertEqual(np.linalg.matrix_rank(D), 1)
        self.assertAlmostEqual(M.edit_cost(D, C), v @ v / M.q_support(k, C, range(n)), places=10)

    def test_schur_gain_matches_difference(self):
        n = 8
        C, k = rand_spd(self.rng, n), self.rng.normal(size=n)
        S = [1, 4, 6]
        for j in (0, 2, 3, 5, 7):
            self.assertAlmostEqual(M.schur_gain(k, C, S, j), M.q_support(k, C, S + [j]) - M.q_support(k, C, S), places=10)

    def test_support_restricted_edit(self):
        n, d = 10, 3
        C, k, v = rand_spd(self.rng, n), self.rng.normal(size=n), self.rng.normal(size=d)
        S = [1, 3, 8]
        K, V = k[:, None], v[:, None]
        D = M.support_edit(K, V, C, S)
        np.testing.assert_allclose(D @ k, v, atol=1e-10)
        self.assertEqual(set(np.nonzero(np.abs(D).sum(0) > 0)[0]), set(S))
        self.assertAlmostEqual(M.edit_cost(D, C), M.support_cost(K, V, C, S), places=10)
        self.assertGreaterEqual(M.support_cost(K, V, C, S), M.optimal_cost(K, V, C) - 1e-12)

    def test_woodbury_matches_dense(self):
        X, lam = self.rng.normal(size=(12, 40)), 0.05
        B = self.rng.normal(size=(40, 3))
        np.testing.assert_allclose(M.woodbury_cinv(X, lam, B), np.linalg.solve(M.second_moment(X, lam), B), atol=1e-9)

    def test_greedy_misses_correlated_pair(self):
        C = np.array([[1, 0, 0], [0, 1, .99], [0, .99, 1.]])
        k = np.array([1.5, 1, -1.])
        Sg, qg = M.select_greedy(k, C, 2)
        Se, qe = M.select_exhaustive(k, C, 2)
        self.assertEqual(sorted(Sg), [0, 1])
        self.assertAlmostEqual(qg, 3.25, places=10)
        self.assertEqual(sorted(Se), [1, 2])
        self.assertAlmostEqual(qe, 200.0, places=8)
        self.assertGreater(qe / qg, 61)
        _, qb = M.select_beam(k, C, 2, width=3)
        self.assertAlmostEqual(qb, qe, places=8)

    def test_beam_never_worse_than_greedy_and_exhaustive_at_full_width(self):
        for seed in range(25):
            rng = np.random.default_rng(seed)
            n = 9
            C, k = rand_spd(rng, n), rng.normal(size=n)
            _, qg = M.select_greedy(k, C, 3)
            _, qb = M.select_beam(k, C, 3, width=6)
            _, qe = M.select_exhaustive(k, C, 3)
            self.assertLessEqual(qb, qe + 1e-12)
            _, qw = M.select_beam(k, C, 3, width=84)       # C(9,3) = 84, so every size-3 set is kept
            self.assertAlmostEqual(qw, qe, places=10)
            self.assertGreaterEqual(qe, qg - 1e-12)

    def test_bf16_rounding(self):
        x = np.array([1.0, 1.00390625, np.nextafter(np.float32(1.00390625), np.float32(2)), -3.14159265, 65504.0, 1e-3], dtype=np.float32)
        y = M.to_bf16(x)
        self.assertEqual(y[0], 1.0)
        self.assertEqual(y[1], 1.0)                         # tie at 1+2^-8 rounds to even (1.0)
        self.assertEqual(y[2], np.float32(1.0078125))
        self.assertTrue(np.all(np.abs(y - x) <= np.abs(x) * 2.0 ** -8 + 1e-30))
        self.assertTrue(np.all((y.view(np.uint32) & 0xFFFF) == 0))
        np.testing.assert_array_equal(M.to_bf16(y), y)

    def test_bf16_matches_torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        x = np.random.default_rng(1).normal(size=20000).astype(np.float32) * np.float32(10.0) ** np.random.default_rng(2).integers(-6, 6, 20000)
        ref = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()
        np.testing.assert_array_equal(M.to_bf16(x), ref)

    def test_eigen_shift_is_exact_rank_one(self):
        n = 6
        A = np.diag([0.9, 0.5, -0.3, 0.2, -0.6, 0.1]) + 0.05 * self.rng.normal(size=(n, n))
        w, R = np.linalg.eig(A)
        i = int(np.argmax(w.real))
        mu, r = w[i].real, R[:, i].real
        Wl, L = np.linalg.eig(A.T)
        l = L[:, int(np.argmin(np.abs(Wl - w[i])))].real
        dA = M.eigen_shift_edit(A, r, l, 0.4, mu)
        self.assertEqual(np.linalg.matrix_rank(dA), 1)
        np.testing.assert_allclose((A + dA) @ r, 0.4 * r, atol=1e-8)
        new = np.linalg.eigvals(A + dA)
        self.assertAlmostEqual(float(np.min(np.abs(new - 0.4))), 0.0, places=7)
        rest_old = sorted(np.delete(w, i), key=lambda z: (z.real, z.imag))
        rest_new = sorted(np.delete(new, int(np.argmin(np.abs(new - 0.4)))), key=lambda z: (z.real, z.imag))
        np.testing.assert_allclose(rest_old, rest_new, atol=1e-7)   # other eigenvalues untouched

    def test_robust_not_worse_than_mean_and_matches_grid(self):
        n, d = 8, 3
        C1, C2 = rand_spd(self.rng, n), rand_spd(self.rng, n) * 5
        K, V = self.rng.normal(size=(n, 1)), self.rng.normal(size=(d, 1))
        Dm = M.min_energy_edit(K, V, 0.5 * (C1 + C2))
        worst_mean = max(M.edit_cost(Dm, C) for C in (C1, C2))
        D, w, worst = M.robust_mixture(K, V, [C1, C2])
        np.testing.assert_allclose(D @ K, V, atol=1e-9)
        self.assertLessEqual(worst, worst_mean + 1e-9)
        grid = max(M.optimal_cost(K, V, t * C1 + (1 - t) * C2) for t in np.linspace(0, 1, 2001))
        self.assertLess(abs(worst - grid) / grid, 5e-3)


if __name__ == "__main__":
    unittest.main()
