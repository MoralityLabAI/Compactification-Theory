import unittest
import numpy as np
from experiments.h_m0_pruner import posterior_update, information_gain, make_episode, run_episode


class Hm0SanityTests(unittest.TestCase):
    def test_posterior_normalizes(self):
        p = posterior_update(np.full(4, 0.25), np.array([0.1, 0.4, 0.9, 0.8]), 1)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertTrue(np.all(p > 0))

    def test_information_gain_nonnegative(self):
        for seed in (4, 100, 10000):
            ep = make_episode(seed)
            for likelihood in ep.likelihoods:
                self.assertGreaterEqual(information_gain(ep.prior, likelihood), 0)

    def test_uninformative_probe_zero_gain(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertLess(information_gain(p, np.array([.6] * 4)), 1e-9)

    def test_duplicated_provenance_requires_deduplication(self):
        # Copied testimony of the *same observation* counts once in a valid
        # provenance-aware update. Repeat Bayes multiplication would overcount.
        p = np.full(4, 0.25)
        q = np.array([.05, .2, .8, .95])
        once = posterior_update(p, q, 1)
        hundred_repetitions_naive = p.copy()
        for _ in range(100):
            hundred_repetitions_naive = posterior_update(hundred_repetitions_naive, q, 1)
        self.assertGreater(100 * (once[3] - .25), 1.0)
        self.assertGreater(float(hundred_repetitions_naive[3]), float(once[3]))
        # This is a probe illustrating a vulnerability; main H-M0 simulator
        # has independent probes and does not implement correlated-source updates.

    def test_budget_and_repeat_probes(self):
        for seed in range(4):
            rec = run_episode('random', make_episode(seed), None)
            self.assertLessEqual(rec['cost'], 5.0)
            self.assertEqual(len(rec['probes']), len(set(rec['probes'])))

if __name__ == '__main__':
    unittest.main()
