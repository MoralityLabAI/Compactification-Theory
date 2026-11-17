"""NOR-37 pre-data exact source, stochastic quota, greedy and witness tests."""
from __future__ import annotations
from fractions import Fraction
from itertools import combinations
from math import comb
from collections import Counter
import unittest

from research.nor37_stochastic_curvature import experiment as x


class NOR37PreDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.past=x.history()
        cls.row=x.new_source(56,512,x.FAMILIES[0],set(cls.past))
        cls.proposal=x.public_proposal(cls.row)
        cls.src=cls.row["full_public_input_sha256"]
        cls.old,cls.cats,cls.u,cls.cu,cls.cv,cls.bu,cls.bv,cls.scores=(
            x.source_geometry(cls.proposal["pred"],cls.proposal["header"],cls.src))

    def test_original_source_receipts_and_registration(self):
        self.assertEqual(len(self.past),156)
        self.assertEqual(len(set(self.past)),156)
        self.assertNotIn(self.src,self.past)
        self.assertEqual(x.REPS,(56,57,58,59))
        self.assertEqual(x.EXPLORE_QUANTA,{"stochastic25":1,"stochastic50":2,"stochastic75":3})
        self.assertEqual(x.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual((x.CAL,x.EVAL,x.K,x.R,x.ATTACKS),(256,512,16,4,8))

    def test_public_native_program_and_no_private_leak(self):
        r=self.row
        self.assertEqual(r["split"],"TEST")
        self.assertTrue(all(key not in r for key in
                            ("program","future","target","attack","witness","audit")))
        self.assertEqual(self.proposal["mask"],r["original_native_public_mask"])
        self.assertEqual(len(self.proposal["pred"]),769)
        self.assertEqual(x.restore_native_target(r,self.proposal["pred"]),
                         self.proposal["pred"])

    def test_all_stochastic_variants_are_exact_quota_matched(self):
        base=x.nor25.policy_schedules(self.old,"test-nor37-base",77)
        for arm in x.ARMS:
            original=tuple(base[(arm,16)])
            quotas=[len(set(original)&c) for c in self.cats]
            for mode in x.EXPLORE_QUANTA:
                new,b=x.new_stochastic_plan(original,self.cats,self.scores,
                     "test-nor37-phase",self.src,arm,77,mode)
                self.assertEqual(len(new),len(set(new)))
                self.assertEqual(len(new),16)
                self.assertEqual([len(set(new)&c) for c in self.cats],quotas)
                self.assertEqual(b["explore_atoms"]+b["targeted_atoms"],
                                 sum(q>0 for q in quotas))
                self.assertEqual((new,b),x.new_stochastic_plan(
                    original,self.cats,self.scores,
                    "test-nor37-phase",self.src,arm,77,mode))

    def test_calibration_and_evaluation_use_independent_phase_salts(self):
        self.assertNotEqual(x.phase_tag(self.src,"cal"),x.phase_tag(self.src,"eval"))
        original=tuple(x.nor25.policy_schedules(
            self.old,"test-nor37-private",4)[("hybrid_50",16)])
        a=x.new_stochastic_plan(original,self.cats,self.scores,"cal",
                                self.src,"hybrid_50",4,"stochastic50")[0]
        b=x.new_stochastic_plan(original,self.cats,self.scores,"eval",
                                self.src,"hybrid_50",4,"stochastic50")[0]
        self.assertNotEqual(a,b)
        # Independent per-atom branch selection must exercise both
        # whole-atom exploration and randomized top-window selection.
        counts=Counter()
        for t in range(60):
            _,v=x.new_stochastic_plan(original,self.cats,self.scores,"cal",
                    self.src,"hybrid_50",t,"stochastic50")
            counts["explore"]+=v["explore_atoms"]
            counts["top"]+=v["targeted_atoms"]
        self.assertGreater(counts["explore"],0)
        self.assertGreater(counts["top"],0)

    def test_policy_components_and_uniform_attack_exact_identity(self):
        self.assertEqual(len(x.POLICIES),10)
        self.assertEqual(len(x.CONTROLS),4)
        for policy in x.POLICIES:
            self.assertEqual(sum(w for _,_,w in x.components(policy)),16)
        for m in range(5,9):
            for k in range(1,4):
                for r in range(1,4):
                    expectation=Fraction(1)-Fraction(comb(m-k,r),comb(m,r))
                    for audited in combinations(range(m),k):
                        tally=sum(bool(set(audited)&set(attack))
                                  for attack in combinations(range(m),r))
                        self.assertEqual(Fraction(tally,comb(m,r)),expectation)

    def test_equal_budget_greedy_calibration_restarter(self):
        m=24
        fake={(mode,arm):[0]*m for mode in x.MODES for arm in x.ARMS}
        for mode,arm in fake:
            for i in range(m):
                fake[(mode,arm)][i]=1<<((i+len(mode)+len(arm))%17)
        for policy in x.POLICIES:
            attacks=[x.greedy_attack(fake,policy,"c"*64,i) for i in range(8)]
            self.assertTrue(all(len(set(a))==4 for a in attacks))
            self.assertTrue(all(all(0<=idx<m for idx in a) for a in attacks))
            self.assertEqual(attacks,[x.greedy_attack(fake,policy,"c"*64,i)
                                      for i in range(8)])

    def test_blind_source_controls_and_true_nonzero_witness(self):
        attacks=x.blind_attacks(self.src,self.cats,self.u,
                               self.cu,self.cv,self.bu,self.bv,769)
        self.assertEqual(set(attacks),set(x.CONTROLS))
        for family,sets in attacks.items():
            self.assertEqual(len(sets),8)
            self.assertTrue(all(len(set(a))==4 for a in sets))
        for a,b in zip(attacks["top_curvature"],attacks["eight_atom_matched"]):
            self.assertEqual([sum(i in atom for i in a) for atom in self.cats],
                             [sum(i in atom for i in b) for atom in self.cats])
        y=tuple(range(12))
        for delta in x.DELTAS:
            self.assertFalse(x.nor24.exact_witness_accepts(
                y,y,(0,3,5),frozenset((3,4,7,9)),delta))
            self.assertTrue(x.nor24.exact_witness_accepts(
                y,y,(0,1,2),frozenset((3,4,7,9)),delta))

    def test_exact_atom_quota_theoretical_invariance(self):
        original=tuple(x.nor25.policy_schedules(
            self.old,"test-nor37-control",8)[("hybrid_50",16)])
        fn=x.nor34.exact_probabilities(self.cats,self.cu,self.cv)
        origcounts=tuple(len(set(original)&c) for c in self.cats)
        target=fn(origcounts,len(set(original)&self.cu),
                  len(set(original)&self.cv))["eight_atom"]
        for mode in x.EXPLORE_QUANTA:
            plan,_=x.new_stochastic_plan(original,self.cats,self.scores,
                   "test-nor37-source",self.src,"hybrid_50",8,mode)
            same=fn(tuple(len(set(plan)&c) for c in self.cats),
                    len(set(plan)&self.cu),
                    len(set(plan)&self.cv))["eight_atom"]
            self.assertEqual(same,target)
        self.assertGreater(x.nor35.ideal_uniform_attack_detect(769),0)


if __name__=="__main__":
    unittest.main()
