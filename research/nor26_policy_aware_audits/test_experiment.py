"""NOR-26 preregistration, independent train/eval streams, greedy attack and finite floor tests."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
import unittest

from research.nor26_policy_aware_audits import experiment as e
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor21_degree_scaling import experiment as nor21


class NOR26Tests(unittest.TestCase):
    def test_frozen_source_and_streams(self):
        self.assertEqual(e.OFFSET,109326000)
        self.assertEqual(e.REPS,(6,7))
        self.assertEqual(e.TRIALS_CAL,256)
        self.assertEqual(e.TRIALS_EVAL,512)
        self.assertEqual(e.ATTACK_REPS,8)
        self.assertEqual(e.RADIUS,4)
        self.assertEqual(e.BUDGET,16)
        self.assertEqual(e.PRIOR_NUMERATORS,(295,246,153,78))
        self.assertEqual(len(e.AUDITORS),7)
        original={n24.source_seed(rep,d,f) for rep in n24.TRAIN_REPS+n24.TEST_REPS
                  for d in n24.DEGREES for f in n24.FAMILIES}
        last={n25.source_seed(rep,d,f) for rep in n25.REPLICATES
              for d in n25.DEGREES for f in n25.FAMILIES}
        new={e.source_seed(rep,d,f) for rep in e.REPS
             for d in e.DEGREES for f in e.FAMILIES}
        self.assertEqual(len(new),12)
        self.assertTrue(new.isdisjoint(original|last))
        self.assertNotEqual(e.phase_digest("a"*64,"calibration"),
                            e.phase_digest("a"*64,"evaluation"))

    def test_bitset_greedy_calibration_objective_and_repeats(self):
        plans=[(0,1),(1,2),(3,4),(1,5),(0,4),(2,5)]
        cov=e.exposure_bitsets(plans,8)
        self.assertEqual(cov[0].bit_count(),2)
        self.assertEqual(cov[1].bit_count(),3)
        self.assertEqual(cov[6],0)
        self.assertEqual(cov[7],0)
        for arm in e.AUDITORS:
            for rep in range(e.ATTACK_REPS):
                chosen,count=e.greedy_attack(cov,"f"*64,arm,rep)
                self.assertEqual(len(chosen),4)
                self.assertEqual(len(set(chosen)),4)
                self.assertEqual(count,
                    sum(bool(set(plan)&set(chosen)) for plan in plans))
                self.assertEqual((chosen,count),
                    e.greedy_attack(cov,"f"*64,arm,rep))
        self.assertNotEqual(e.fresh_attack("b"*64,769,0),
                            e.fresh_attack("b"*64,769,1))

    def test_ideal_two_stage_floor_exhaustive(self):
        for m in range(7,11):
            for t in (1,2):
                for u in (1,2,3):
                    for r in (1,2,3):
                        total=comb(m-t,u)
                        floor=Fraction(1)-Fraction(comb(m-t-r,u),total)
                        seen=[]
                        for attack in combinations(range(m),r):
                            hits=0
                            for probe in combinations(range(t,m),u):
                                hits+=bool(set(attack)&(set(range(t))|set(probe)))
                            seen.append(Fraction(hits,total))
                        self.assertEqual(min(seen),floor)
        self.assertEqual(e.theoretical_floor(769),n25.blind_floor(769,16,8,4))
        self.assertEqual(e.theoretical_floor(1537),n25.blind_floor(1537,16,8,4))
        self.assertGreater(e.theoretical_floor(1537),Fraction(2,100))

    def test_one_fresh_public_only_source_parity(self):
        row=e.new_public_source(6,512,e.FAMILIES[0],set())
        self.assertEqual(row["replicate"],6)
        self.assertEqual(row["split"],"TEST")
        self.assertTrue(all(s not in row for s in
                            ("program","future","witness","attack")))
        public,header=n24.public_only_case(row)
        self.assertIsNone(public.program)
        self.assertEqual(public.future,())
        self.assertEqual(nor21.native_mask(public,header,512),
                         row["original_public_accept_mask"])
        proposal=n24.public_proposal(row)
        self.assertEqual(len(proposal["predicted"]),769)


if __name__=="__main__":
    unittest.main()
