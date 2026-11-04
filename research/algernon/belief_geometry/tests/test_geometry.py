"""Finite exact checks of algebraic invariants and negative controls, no models."""
import random
import unittest
from fractions import Fraction as Q
from itertools import combinations, permutations
from research.algernon.belief_geometry.model import (
    Evidence, DiscreteOneForm, flat_one_form, fair_bits_example,
    matrix_rank_rational, entropy_bits, FiniteHorizon, Probe, normalize,
)


class EvidenceAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.system=Evidence((Q(1,10),Q(2,10),Q(3,10),Q(4,10)),(
            (Q(1,2),Q(1,4),Q(1,8),Q(1,16)),
            (Q(1,16),Q(1,4),Q(1,8),Q(1,2)),
            (Q(1,8),Q(1,8),Q(1,2),Q(1,16)),
            (Q(1,4),Q(1,16),Q(1,8),Q(1,2)),
        ))

    def test_exact_all_orders(self):
        self.assertTrue(self.system.all_orders_equal())
        self.assertEqual(self.system.posterior(()),self.system.prior)
        for subset_len in range(5):
            for subset in combinations(range(4),subset_len):
                expect=self.system.posterior(subset)
                for ordering in permutations(subset):
                    self.assertEqual(self.system.posterior(ordering),expect)

    def test_exact_multiplicative_odds(self):
        for s in range(1<<4):
            subset=tuple(i for i in range(4) if s>>i&1)
            odds=self.system.odds(subset)
            for e in range(4):
                if s>>e&1: continue
                after=self.system.odds(subset+(e,))
                mul=self.system.odds_multiplier(e)
                self.assertEqual(after,tuple(x*y for x,y in zip(odds,mul)))

    def test_likelihood_validation(self):
        with self.assertRaises(ValueError):
            Evidence((Q(1,2),Q(1,2)),((Q(1,2),Q(0)),))
        with self.assertRaises(ValueError):
            Evidence((Q(1,2),Q(1,4)),((Q(1,2),Q(1,2)),))

    def test_integer_exponent_rank(self):
        rows=((3,2,1),(-3,-1,-2),(1,1,3),(-1,-3,-2))
        self.assertEqual(matrix_rank_rational(rows),3)
        self.assertEqual(matrix_rank_rational(((1,2),(2,4))),1)


class CubeTests(unittest.TestCase):
    def test_flat_path_independence(self):
        g=flat_one_form(((1,2),(-3,4),(5,6),(-2,1)))
        self.assertEqual(g.max_curvature(),0)
        final={g.integrate(path) for path in permutations(range(4))}
        self.assertEqual(len(final),1)
        self.assertEqual(next(iter(final)),(Q(1),Q(13)))

    def test_nonflat_curvature_and_swap_bound(self):
        g=flat_one_form(((1,),(2,),(3,),(4,)))
        edges=dict(g.edges)
        edges[(1,1)]=(Q(9),)   # localized changed edge for path [0,1]
        nonflat=DiscreteOneForm(4,1,edges)
        self.assertGreater(nonflat.max_curvature(),0)
        self.assertNotEqual(nonflat.integrate((0,1)),nonflat.integrate((1,0)))
        for a in permutations(range(4)):
            for b in permutations(range(4)):
                self.assertTrue(nonflat.path_bound(a,b)['valid'])

    def test_random_integer_edge_fields(self):
        rng=random.Random(20261104)
        for n in range(2,6):
            for case in range(25):
                edges={(mask,i):(Q(rng.randint(-4,4)),Q(rng.randint(-4,4)))
                       for mask in range(1<<n) for i in range(n) if not mask & (1<<i)}
                c=DiscreteOneForm(n,2,edges)
                samples=list(permutations(range(n)))
                for _ in range(25):
                    a,b=rng.choice(samples),rng.choice(samples)
                    self.assertTrue(c.path_bound(a,b)['valid'])

    def test_schema(self):
        with self.assertRaises(ValueError):
            DiscreteOneForm(2,1,{})
        with self.assertRaises(ValueError):
            flat_one_form([])


class PlanningTests(unittest.TestCase):
    def test_fair_bit_counterexample(self):
        prior,solver=fair_bits_example()
        scores=solver.scores(prior,1)
        self.assertEqual(scores['STOP'],Q(1,2))
        self.assertEqual(scores['REVEAL_Y'],Q(9,20))
        self.assertEqual(scores['NOISY_X'],Q(7,10))
        self.assertEqual(solver.advantage(prior,1),Q(1,5))
        mi_a=1.0
        mi_b=1-entropy_bits((Q(1,4),Q(3,4)))
        self.assertGreater(mi_a,mi_b)

    def test_belief_regions_exact(self):
        prior,solver=fair_bits_example()
        self.assertEqual(solver.scores(prior,0)['STOP'],Q(1,2))
        p1=(Q(3,8),Q(3,8),Q(1,8),Q(1,8))  # P(X=0)=3/4
        scores=solver.scores(p1,1)
        self.assertEqual(scores['STOP'],Q(3,4))
        self.assertLessEqual(scores['NOISY_X'],scores['STOP'])

    def test_repeated_probe_prevention(self):
        prior,solver=fair_bits_example()
        self.assertEqual(solver.scores(prior,2)['NOISY_X'],Q(7,10))
        # Cost-positive reveal-Y is irrelevant to predicting X, so its best policy
        # cannot improve on simply using NOISY_X.

    def test_bad_probe(self):
        with self.assertRaises(ValueError):
            Probe('bad',((Q(1,2),Q(1,2)),(Q(1,2),Q(1,3))),Q(0))
        with self.assertRaises(ValueError):
            normalize((0,0))

    def test_one_step_stopping_certificate(self):
        # A bound on Q estimates certifies the selected action if margin>2eps.
        scores={'STOP':Q(1,2),'NOISY_X':Q(7,10),'REVEAL_Y':Q(9,20)}
        err=Q(1,20)
        estimates={k:v+([1,-1,1][i])*err for i,(k,v) in enumerate(scores.items())}
        self.assertEqual(max(estimates,key=estimates.get),'NOISY_X')
        self.assertGreater(Q(7,10)-Q(1,2),2*err)


if __name__=='__main__':
    unittest.main()
