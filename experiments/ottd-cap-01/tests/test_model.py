"""Synthetic-world invariants and exact rational oracle checks."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (Action, Route, Scenario, Scout, Greedy, advance, initial,
                   observe, legal_actions, run_episode, verify_receipt,
                   seeded_world, openttd_style_score)
from oracle import exact_bayesian_value, clairvoyant_mean, policy_mean
from bench import exact_fixture, paired_sweep


class WorldTests(unittest.TestCase):
    def test_accounting_and_delayed_revenue(self):
        s = Scenario((Route("coal", 3, 8, -10),), 3, 3)
        after_build = advance(s, initial(s), 1, Action("build", 0))
        self.assertEqual((after_build.cash, after_build.operating_profit,
                          after_build.capital_spent), (0, 0, 3))
        after_wait = advance(s, after_build, 1, Action("wait"))
        self.assertEqual((after_wait.cash, after_wait.operating_profit), (8, 8))
        loss = advance(s, after_build, 0, Action("wait"))
        self.assertEqual((loss.cash, loss.operating_profit), (-10, -10))

    def test_survey_ephemeral_and_private(self):
        s = Scenario((Route("a", 5, 8, -2),), 3, 20)
        self.assertEqual(observe(initial(s)), observe(initial(s)))
        high = advance(s, initial(s), 1, Action("survey", 0))
        low = advance(s, initial(s), 0, Action("survey", 0))
        self.assertEqual(high.latest_signal, (0, 1))
        self.assertEqual(low.latest_signal, (0, 0))
        self.assertIsNone(advance(s, high, 1, Action("wait")).latest_signal)

    def test_illegal_build_and_horizon(self):
        s = Scenario((Route("a", 5, 8, -2),), 2, 5)
        self.assertIn(Action("build", 0), legal_actions(s, initial(s)))
        x = advance(s, initial(s), 1, Action("build", 0))
        with self.assertRaises(ValueError):
            advance(s, x, 1, Action("build", 0))
        x = advance(s, x, 1, Action("wait"))
        self.assertEqual(legal_actions(s, x), ())
        with self.assertRaises(ValueError):
            advance(s, x, 1, Action("wait"))

    def test_scoring_ties_and_operating_only(self):
        self.assertAlmostEqual(openttd_style_score(100, (0, 100), 100), 0.8)
        self.assertAlmostEqual(openttd_style_score(100, (0, 1), 100), 1.0)
        self.assertAlmostEqual(openttd_style_score(-1, (0,), 100), 0)
        with self.assertRaises(ValueError):
            openttd_style_score(0, ())

    def test_hash_chain_and_json_replay(self):
        s, mask = seeded_world(101, 3, 9)
        original = run_episode(s, mask, Scout(6))
        self.assertTrue(verify_receipt(original))
        self.assertTrue(verify_receipt(json.loads(json.dumps(original))))
        tampered = deepcopy(original)
        tampered["trace"][0]["after"]["cash"] += 100
        self.assertFalse(verify_receipt(tampered))

    def test_memory_and_seed_reproduction(self):
        s, mask = seeded_world(202, 3, 9)
        self.assertEqual((s, mask), seeded_world(202, 3, 9))
        for bits in [0, 1, 2, 3, 6, 9]:
            record = run_episode(s, mask, Scout(bits))
            self.assertLessEqual(record["peak_memory_bits"], bits)
            self.assertEqual(record["steps"], s.horizon)
        self.assertEqual(run_episode(s, mask, Greedy()),
                         run_episode(s, mask, Greedy()))


class OracleTests(unittest.TestCase):
    def test_hand_calculated_survey_option(self):
        s = Scenario((Route("one", 3, 8, -10),), 3, 3)
        # First month survey, second month build only if good, third earns 8.
        self.assertEqual(exact_bayesian_value(s), Fraction(5, 2))
        self.assertEqual(clairvoyant_mean(s), Fraction(13, 2))

    def test_information_bound(self):
        s, _ = seeded_world(13, 2, 5)
        optimum = exact_bayesian_value(s)
        self.assertLessEqual(policy_mean(s, Greedy), optimum)
        self.assertLessEqual(policy_mean(s, lambda: Scout(3)), optimum)
        self.assertLessEqual(optimum, clairvoyant_mean(s))

    def test_strict_exact_oracle_scope(self):
        s, _ = seeded_world(13, 4, 8)
        with self.assertRaises(ValueError):
            exact_bayesian_value(s)

    def test_sweep_pairs_worlds(self):
        data = paired_sweep([1, 2], 3, 8, 80, [0, 3, 6, 9])
        self.assertEqual(len(data["records"]), 10)
        self.assertEqual(len({r["quality_mask_evaluator_only"]
                              for r in data["records"] if r["seed"] == 1}), 1)
        self.assertEqual(exact_fixture(1, 2, 5, 80, 3)["mode"],
                         "finite_exact_synthetic")


if __name__ == "__main__":
    unittest.main()
