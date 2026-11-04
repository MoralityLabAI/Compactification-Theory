import sys, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase_e1_recurrent as E


def make_W(rng, n=12, lead=0.9):
    u = rng.normal(size=n)
    return 0.3 * rng.normal(size=(n, n)) / np.sqrt(n) + lead * np.outer(u, u) / (u @ u)


class PhaseE1(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_rosters_are_disjoint_and_deterministic(self):
        r1, r2 = E.roster(), E.roster()
        self.assertEqual(r1["sha256"], r2["sha256"])
        Xa, sa = E.gen(E.SPLITS["calib"]["lo"], 8, 20)
        Xb, sb = E.gen(E.SPLITS["calib"]["lo"], 8, 20)
        Xt, st = E.gen(E.SPLITS["test"]["lo"], 8, 20)
        self.assertTrue(torch.equal(Xa, Xb) and torch.equal(sa, sb))
        self.assertFalse(torch.equal(sa, st))
        self.assertEqual(E.data_hash(Xa, sa), E.data_hash(Xb, sb))
        s = list(E.SPLITS.values())
        self.assertTrue(s[0]["hi"] < s[1]["lo"] and s[1]["hi"] < s[2]["lo"])

    def test_eigen_edit_moves_mean_jacobian_eigenvalue_only(self):
        W, Dbar = make_W(self.rng), self.rng.uniform(0.8, 1.0, 12)
        dW, m, mu_t = E.eig_edit(W, Dbar, "J", lambda mu: 0.95 * mu)
        self.assertAlmostEqual(mu_t, 0.95 * m["mu"])
        m2 = E.mode_of(W + dW, Dbar)
        self.assertAlmostEqual(m2["mu"], 0.95 * m["mu"], places=9)
        ev0 = np.sort_complex(np.delete(m["ev"], m["i"]))
        ev1 = np.sort_complex(np.delete(m2["ev"], m2["i"]))
        np.testing.assert_allclose(ev1, ev0, atol=1e-8)
        self.assertEqual(np.linalg.matrix_rank(dW), 1)

    def test_controls_match_norm_and_support(self):
        rho, n = 0.37, 12
        D = E.rand_dense(self.rng, n, rho)
        self.assertAlmostEqual(np.linalg.norm(D), rho)
        self.assertEqual(np.linalg.matrix_rank(D), 1)
        rows, cols = np.array([1, 4, 5]), np.array([0, 2, 7])
        S = E.rand_support(self.rng, rows, cols, n, rho)
        self.assertAlmostEqual(np.linalg.norm(S), rho)
        mask = np.zeros((n, n), bool)
        mask[np.ix_(rows, cols)] = True
        self.assertEqual(np.abs(S[~mask]).max(), 0.0)

    def test_transient_growth_and_condition_of_nonnormal_matrix(self):
        W = np.array([[0.5, 3.0], [0.0, 0.4]])
        sp = E.spectrum(W, np.ones(2), K=50)
        self.assertAlmostEqual(sp["rho"], 0.5)
        self.assertGreater(sp["transient_max"], 2.0)
        self.assertGreater(sp["kappa_lead"], 3.0)

    def test_lyapunov_matches_log_of_leading_eigenvalue_in_linear_regime(self):
        Q, _ = np.linalg.qr(self.rng.normal(size=(8, 8)))
        W = Q @ np.diag([0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]) @ Q.T
        h = torch.zeros(16, E.CFG["T_long"], 8)                           # tanh' = 1: Jacobian is W
        self.assertAlmostEqual(E.lyapunov(h, W)["lyap160"], np.log(0.9), delta=0.01)

    def test_bootstrap_ci_is_deterministic_and_brackets_mean(self):
        d = np.random.default_rng(1).normal(0.2, 1.0, 300)
        a, b = E.paired_boot(d), E.paired_boot(d)
        self.assertEqual(a, b)
        self.assertLess(a[1], a[0])
        self.assertLess(a[0], a[2])
        self.assertAlmostEqual(a[0], d.mean())

    def test_training_edit_and_gradient_paths_run(self):
        net, curve = E.train(steps=120)
        self.assertLess(curve[-1][1], 0.5 * curve[0][1])
        X, s = E.gen(E.SPLITS["calib"]["lo"], 64, E.CFG["T_long"])
        base, sq, h = E.evaluate(net, X, s)
        self.assertEqual(sq["sq160"].shape, (64,))
        W = E.get_W(net)
        Dbar = (1 - h[:, 1:40] ** 2).mean((0, 1)).double().numpy()
        dW, m, _ = E.eig_edit(W, Dbar, "J", lambda mu: mu - 0.01)
        e0 = E.edited(net, dW, False, X, s)
        self.assertEqual(e0.alpha, 1.0)
        np.testing.assert_allclose(E.get_W(e0), W + dW, atol=1e-6)
        e1 = E.edited(net, dW, True, X, s)                                  # alpha is the least-squares gain at t=40
        with torch.no_grad():
            y = e0(X[:, :40])[1][:, -1]
        self.assertAlmostEqual(e1.alpha, float((y * s).sum() / (y ** 2).sum()), places=4)
        g = E.grad_edits(net, X, s, 1e-4)
        self.assertAlmostEqual(np.linalg.norm(g["grad_full"]), 1e-4, places=9)
        self.assertEqual(np.linalg.matrix_rank(g["grad_rank1"]), 1)
        after, _, _ = E.evaluate(E.apply_edit(net, g["grad_full"]), X, s)
        self.assertLess(after["mse160"], base["mse160"])                    # steepest-descent step lowers its own loss


if __name__ == "__main__":
    unittest.main()
