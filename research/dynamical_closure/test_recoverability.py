"""Exact finite-state risk transfer checks and held-out intervention tests."""
import unittest
from fractions import Fraction as Q
from itertools import product

from closure import partitions
from recoverability_experiment import (STATES, build_system, code_path,
                                       fit_transition, fitted_mse, run)
from task_closure import bayes_risk, run_word


class RecoverabilityTests(unittest.TestCase):
    def test_density_ratio_bound_for_all_small_partitions(self):
        # min_x nu(x)/mu(x) R_mu <= R_nu <= max_x nu(x)/mu(x) R_mu.
        for n in (2, 3):
            for code in partitions(n):
                for target in product((0, 1), repeat=n):
                    mu = tuple(range(1, n + 1))
                    for nu in product((1, 2), repeat=n):
                        ratios = tuple(Q(nu[x], sum(nu)) / Q(mu[x], sum(mu))
                                       for x in range(n))
                        self.assertLessEqual(min(ratios) * bayes_risk(code, target, mu),
                                             bayes_risk(code, target, nu))
                        self.assertLessEqual(bayes_risk(code, target, nu),
                                             max(ratios) * bayes_risk(code, target, mu))

    def test_threshold_bracket_under_two_sided_shift(self):
        system, names = build_system()
        words = ((names.index("reveal_b"),), (names.index("flip_b"), names.index("reveal_b")))
        mu = (1,) * len(STATES)
        nu = (1, 1, 2, 3, 2, 1, 1, 2)
        ratios = tuple(Q(nu[x], sum(nu)) / Q(mu[x], sum(mu)) for x in range(len(mu)))
        m, big_m = min(ratios), max(ratios)
        def risks(weights):
            return [max(bayes_risk(c, tuple(system.task[run_word(system, x, w)]
                                            for x in range(len(STATES))), weights)
                        for w in words) for c in code_path()]
        observed, deployed = risks(mu), risks(nu)
        delta = Q(1, 16)
        first = lambda rs, d: next((k for k, r in enumerate(rs) if r > d), float("inf"))
        self.assertLessEqual(first(observed, delta / big_m), first(deployed, delta))
        self.assertLessEqual(first(deployed, delta), first(observed, delta / m))

    def test_support_change_and_target_change_need_assumptions(self):
        self.assertEqual(bayes_risk((0, 0), (0, 1), (1, 0)), 0)
        self.assertEqual(bayes_risk((0, 0), (0, 1), (1, 1)), Q(1, 4))
        # Even mu=nu (M=1) cannot transfer across unrelated action targets.
        self.assertEqual(bayes_risk((0, 0), (0, 0)), 0)
        self.assertEqual(bayes_risk((0, 0), (0, 1)), Q(1, 4))

    def test_population_and_fitted_compositions(self):
        result = run()
        self.assertEqual(result["states"], 8)
        self.assertEqual(result["training"]["task_only_training_risk"], "0")
        self.assertEqual(result["pilot_reveal_b"]["oracle_bayes_risk"],
                         ["0", "0", "1/4", "1/4"])
        self.assertEqual(result["pilot_reveal_b"]["threshold_k"], 2)
        self.assertEqual(result["closed_on_fitted_actions"], [True, True, False, False])
        for item in result["heldout_compositions"].values():
            self.assertEqual(item["oracle_bayes_risk"], ["0", "0", "1/4", "1/4"])
            self.assertEqual(item["fitted_composition_mse"], ["0", "0", "1/4", "1/4"])
            self.assertEqual(item["threshold_k"], 2)
        self.assertEqual(result["unseen_action_reveal_c"]["oracle_bayes_risk"],
                         ["0", "1/4", "1/4", "1/4"])
        self.assertEqual(result["unseen_action_reveal_c"]["threshold_k"], 1)
        self.assertNotIn("fitted_composition_mse", result["unseen_action_reveal_c"])

    def test_fitted_transition_is_exact_for_closed_code(self):
        system, names = build_system()
        codes = code_path()
        for name in ("identity", "flip_y", "flip_b", "reveal_b"):
            action = names.index(name)
            table = fit_transition(codes[1], system.transitions[action])
            for x, z in enumerate(codes[1]):
                self.assertEqual(table[z], codes[1][run_word(system, x, (action,))])
        word = (names.index("flip_b"), names.index("reveal_b"))
        for c in codes:
            tables = {i: fit_transition(c, system.transitions[i]) for i in word}
            self.assertEqual(fitted_mse(system, c, tables, word),
                             bayes_risk(c, tuple(system.task[run_word(system, x, word)]
                                                 for x in range(len(STATES)))))


if __name__ == "__main__":
    unittest.main()
