"""NOR-31 finite policy-class quotient, training constraints and witness tests."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations,permutations
from math import comb
import unittest

from research.nor31_class_constrained_minimax import train as t
from research.nor31_class_constrained_minimax import experiment as n
from research.nor21_degree_scaling import experiment as nor21


class NOR31Tests(unittest.TestCase):
    def test_exact_uniform_law_after_fixed_ranking_permutation(self):
        for m in range(2,7):
            ranks=tuple(reversed(range(m)))
            trials=list(permutations(range(m)))
            for k in range(1,min(4,m)+1):
                shuffled=Counter(tuple(permutation[i] for i in ranks[:k])
                                 for permutation in trials)
                uniform=Counter(permutation[:k] for permutation in trials)
                self.assertEqual(shuffled,uniform,(m,k))
                self.assertEqual(len(uniform),comb(m,k)*__import__('math').factorial(k))

    def test_histories_are_pinned_and_minimax_has_real_risk_mass(self):
        receipt=t.freeze_train()
        self.assertEqual(receipt["schema"],t.SCHEMA)
        self.assertEqual(receipt["training_public_source_count"],24)
        self.assertEqual(len(set(receipt["historical_nor26_public_hashes"]+
                                 receipt["historical_nor30_public_hashes"])),24)
        self.assertFalse(receipt["test_source_information_used"])
        w=receipt["fitted_constrained"]["weights"]
        self.assertEqual(len(w),6)
        self.assertEqual(sum(w),16)
        self.assertTrue(2<=w[0]<=12)
        self.assertGreater(sum(w[1:]),0)
        self.assertGreaterEqual(receipt["fitted_constrained"]["historical_in_template_rate"],.18)
        self.assertEqual(receipt["fitted_constrained"],t.fit(
            receipt["historical_nor26_matrix_integer_hits"],
            receipt["historical_nor30_template_weighted_numerators"],True,True))

    def test_six_class_optimizer_tie_rule_and_constraint(self):
        matrix=[[0,0,0,0,0,20] for _ in range(7)]
        templates=[5,5,5,20,20,20]
        # Low floor for tiny toy universe: test the exact optimizer with
        # admissible raw count units, rather than changing registered 18%.
        r=t.fit(matrix,[12*4096*10]*6,True,True)
        self.assertEqual(sum(r["weights"]),16)
        self.assertEqual(r["weights"][0],2)
        self.assertEqual(r["weights"][-1],14)
        self.assertTrue(r["candidate_grid_eligible"]>0)

    def test_source_separation_and_native_parity_before_hidden_truth(self):
        hist=t.freeze_train()
        seen=set(hist["historical_nor26_public_hashes"]+
                 hist["historical_nor30_public_hashes"])
        row=n.new_public_source(24,512,n.FAMILIES[0],seen)
        self.assertEqual(row["split"],"TEST")
        self.assertNotIn(row["full_public_input_sha256"],
                         hist["historical_nor26_public_hashes"])
        self.assertNotIn(row["full_public_input_sha256"],
                         hist["historical_nor30_public_hashes"])
        for private in ("program","future","target","attack","witness"):
            self.assertNotIn(private,row)
        proposal=n.public_proposal(row)
        self.assertEqual(proposal["mask"],row["exact_native_public_mask"])
        self.assertEqual(len(proposal["pred"]),769)
        self.assertEqual(tuple(n.private_future_after_freeze(row,proposal["pred"])),
                         tuple(proposal["pred"]))

    def test_blind_and_fully_informed_exact_integer_witness(self):
        pred=(1,3,5,7,9,11,13)
        attack=(2,5)
        self.assertTrue(n.detect(pred,pred,(0,2),attack))
        self.assertFalse(n.detect(pred,pred,(0,1,4),attack))
        self.assertTrue(n.detect(pred,pred,(5,),attack))
        self.assertFalse(n.detect(pred,pred,(),attack))
        self.assertEqual(n.SHIFTS,(1,4294049777))

    def test_greedy_attack_schedules_and_reproducible_cat_draw(self):
        m=20
        cover={a:[0]*m for a in n.LEGACY_ATTACKS}
        for a in n.LEGACY_ATTACKS:
            for i in range(m):
                cover[a][i]=1 << ((i+len(a))%16)
        weights=(2,0,1,1,4,8)
        for name in n.ATTACKERS:
            r=n.greedy_attack(cover,weights,"a"*64,name,0)
            self.assertEqual(len(r),4)
            self.assertEqual(len(set(r)),4)
            self.assertEqual(r,n.greedy_attack(cover,weights,"a"*64,name,0))
        picks=[n.choose_mixed(weights,"b"*64,i) for i in range(200)]
        self.assertTrue(set(picks).issubset(n.ARMS))
        self.assertGreater(len(set(picks)),1)
        self.assertNotEqual(n.phase_tag("b"*64,"calibration"),
                            n.phase_tag("b"*64,"evaluation"))

    def test_ideal_source_conditional_mix_floor_is_positive(self):
        for m in (769,1537):
            self.assertEqual(n.ideal_floor(m,"static_prior"),0)
            self.assertEqual(n.ideal_floor(m,"posterior"),0)
            self.assertEqual(n.ideal_floor(m,"uniform"),
                             Fraction(1)-Fraction(comb(m-4,16),comb(m,16)))
            w=(2,0,1,1,4,8)
            self.assertGreater(sum(Fraction(p,16)*n.ideal_floor(m,arm)
                                   for p,arm in zip(w,n.ARMS)),0)
        m,k,r=8,3,2
        for attack in combinations(range(m),r):
            total=comb(m,k)
            detected=sum(bool(set(attack)&set(probes))
                         for probes in combinations(range(m),k))
            self.assertEqual(Fraction(detected,total),
                Fraction(1)-Fraction(comb(m-r,k),comb(m,k)))


if __name__=="__main__":
    unittest.main()
