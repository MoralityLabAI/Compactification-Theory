"""NOR-25 source split, audit support, deterministic policies and exact blind bound."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
import unittest

from research.nor25_robust_audits import experiment as e
from research.nor24_adaptive_audits import experiment as n
from research.rrc_compactification import compound25 as corpus
from research.nor21_degree_scaling import experiment as nor21


class NOR25RobustAuditTests(unittest.TestCase):
    def test_frozen_params_prior_and_source_disjointness(self):
        self.assertEqual(e.SOURCE_OFFSET, 99325000)
        self.assertEqual(e.REPLICATES, (4, 5))
        self.assertEqual(e.SOURCE_COUNT, 12)
        self.assertEqual(e.TRIALS, 1024)
        self.assertEqual(e.PRIOR_NUMERATORS, (295, 246, 153, 78))
        self.assertEqual(sum(e.PRIOR_NUMERATORS), e.PRIOR_DENOMINATOR)
        self.assertEqual(e.reserve_size("hybrid_25",16),4)
        self.assertEqual(e.reserve_size("hybrid_50",16),8)
        self.assertEqual(e.reserve_size("hybrid_75",16),12)
        self.assertEqual(len(set(e.source_seed(r,d,f) for r in e.REPLICATES
                                 for d in e.DEGREES for f in e.FAMILIES)),12)
        original = {n.source_seed(r,d,f)
                    for r in n.TRAIN_REPS+n.TEST_REPS
                    for d in n.DEGREES for f in n.FAMILIES}
        fresh = {e.source_seed(r,d,f) for r in e.REPLICATES
                 for d in e.DEGREES for f in e.FAMILIES}
        self.assertTrue(fresh.isdisjoint(original))

    def test_hypergeometric_minimax_exact_small_universe(self):
        for m in range(4, 9):
            for k in range(1, min(5,m)+1):
                for u in range(k+1):
                    t = k-u
                    for radius in range(1,min(3,m-t)+1):
                        T = set(range(t))
                        remaining = tuple(range(t,m))
                        samples = tuple(combinations(remaining,u))
                        assert len(samples) == comb(m-t,u)
                        worst = Fraction(1)
                        for A in combinations(range(m),radius):
                            attack = set(A)
                            hits = sum(bool((T|set(R))&attack) for R in samples)
                            rate = Fraction(hits,len(samples))
                            worst = min(worst,rate)
                            if not (attack&T):
                                self.assertEqual(rate,e.blind_floor(m,k,u,radius))
                            else:
                                self.assertEqual(rate,1)
                        self.assertEqual(worst,e.blind_floor(m,k,u,radius))

    def test_hybrid_uniform_reserve_independent_seed_and_exact_sizes(self):
        m=769
        static_order=tuple(range(64))
        for policy in e.MIXED_FRACTIONS:
            for budget in e.BUDGETS:
                x=e.hybrid_schedule(static_order,m,"c"*64,policy,budget,17)
                y=e.hybrid_schedule(static_order,m,"c"*64,policy,budget,17)
                self.assertEqual(x,y)
                self.assertEqual(len(x),budget)
                self.assertEqual(len(set(x)),budget)
                t=budget-e.reserve_size(policy,budget)
                self.assertEqual(x[:t], static_order[:t])
                self.assertTrue(set(x[t:]).isdisjoint(x[:t]))
        self.assertGreater(e.blind_floor(769,16,8,4),Fraction(2,100))
        self.assertGreater(e.blind_floor(1537,16,8,4),Fraction(2,100))
        self.assertEqual(e.blind_floor(769,16,0,4),0)
        self.assertEqual(e.blind_floor(769,16,16,4),
                         Fraction(1)-Fraction(comb(769-4,16),comb(769,16)))

    def test_all_equal_budget_arms_and_informed_evasion(self):
        y=tuple(i*i if i%7 == 0 else -i for i in range(769))
        h=corpus.Header("I",80,32,1)
        supports=n.supports(y,h)
        allplans=e.policy_schedules(supports,"f"*64,12)
        self.assertEqual(allplans,e.policy_schedules(supports,"f"*64,12))
        for budget in e.BUDGETS:
            for arm in e.ARMS:
                for kind in e.ALL_KINDS:
                    p=allplans[(arm,kind,budget)] if arm=="oracle_support" \
                      else allplans[(arm,budget)]
                    self.assertEqual(len(set(p)),budget)
                    self.assertTrue(all(0<=i<769 for i in p))
                    for radius in e.RADIUS:
                        attack=n.informed_attack(p,supports[kind],radius)
                        self.assertEqual(len(attack),radius)
                        self.assertTrue(set(p).isdisjoint(attack))
                        self.assertFalse(e.exact_detection(y,y,p,attack,1))
                        self.assertFalse(e.exact_detection(y,y,p,attack,4294049777))

    def test_original_native_verifier_on_one_new_public_source(self):
        seen=set()
        row=e.generate_source(4,512,e.FAMILIES[0],seen)
        self.assertEqual(row["split"],"TEST")
        self.assertEqual(len(seen),1)
        self.assertTrue(all(s not in row for s in ("program","future","attack")))
        c,h=n.public_only_case(row)
        self.assertIsNone(c.program)
        self.assertEqual(c.future,())
        self.assertEqual(nor21.native_mask(c,h,512),
                         row["original_public_accept_mask"])
        prop=n.public_proposal(row)
        self.assertEqual(prop["mask"],row["original_public_accept_mask"])
        self.assertEqual(len(prop["predicted"]),769)


if __name__=="__main__":
    unittest.main()
