import random
import unittest

from research.algernon.metacognition.decision_margin import Decision,bound_from_steps


class DecisionMarginTests(unittest.TestCase):
    def test_exact(self):
        a=Decision(("stop","probe"),(.7,.8),(.7,.8)).audit()
        self.assertEqual(a["single_decision_regret"],0)
        self.assertTrue(a["margin_certified"])

    def test_tight_bound(self):
        a=Decision(("stop","probe"),(0.,2.),(1.,1.)).audit()
        self.assertEqual(a["single_decision_regret"],2.)
        self.assertEqual(a["q_sup_error"],1.)

    def test_random_bruteforce(self):
        rng=random.Random(20261104)
        for _ in range(10000):
            oracle=[rng.uniform(-1,1) for _ in range(5)]
            epsilon=rng.uniform(0,.4)
            estimated=[v+rng.uniform(-epsilon,epsilon) for v in oracle]
            Decision(tuple(map(str,range(5))),tuple(oracle),tuple(estimated)).audit()

    def test_episode_bound(self):
        self.assertAlmostEqual(bound_from_steps([.01,.02,.03]),.12)
        with self.assertRaises(ValueError):
            bound_from_steps([-.1])

    def test_bad_schema(self):
        with self.assertRaises(ValueError):
            Decision(("stop","stop"),(.1,.2),(.1,.2))


if __name__=="__main__":
    unittest.main()
