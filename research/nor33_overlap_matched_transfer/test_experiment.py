"""NOR-33 exact matching, public-only parity and two-block finite controls."""
from __future__ import annotations

from itertools import combinations, product
from math import comb
from fractions import Fraction
import unittest

from research.nor33_overlap_matched_transfer import experiment as n
from research.nor24_adaptive_audits import experiment as old24


class NOR33Controls(unittest.TestCase):
    def test_frozen_original_source_corpus_and_nor31_weights(self):
        history=n.past_source_digests()
        self.assertEqual(len(history),60)
        self.assertEqual(len(set(history)),60)
        self.assertEqual(n.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual(n.SUPPORT_FAMILIES,(
            "central","stride17","low_abs","curvature",
            "anti_prior","hash","stratified_random"))
        self.assertEqual(n.STRATA,(0,1,2,3,4))
        self.assertEqual((n.ATTACKS,n.TRIALS,n.K,n.R),(8,512,16,4))

    def test_new_complete_public_input_does_not_expose_target(self):
        seen=set(n.past_source_digests())
        row=n.generate_source(40,512,n.FAMILIES[0],seen)
        self.assertEqual(row["replicate"],40)
        self.assertNotIn("program",row)
        self.assertNotIn("future",row)
        self.assertNotIn("target",row)
        self.assertNotIn("attack",row)
        self.assertNotIn("witness",row)
        self.assertEqual(row["split"],"TEST")
        proposal=n.public_proposal(row)
        self.assertEqual(proposal["mask"],row["native_public_mask"])
        self.assertEqual(len(proposal["pred"]),769)
        target=n.private_truth_postchoice(row,proposal["pred"])
        self.assertEqual(target,proposal["pred"])

    def test_rankings_supports_and_every_attack_exactly_overlap_matched(self):
        seen=set(n.past_source_digests())
        row=n.generate_source(40,512,n.FAMILIES[0],seen)
        p=n.public_proposal(row)
        historical=old24.supports(p["pred"],p["header"])
        u=n.risk_union(historical)
        orders=n.full_rankings(
            p["pred"],p["header"],row["full_public_input_sha256"],historical)
        self.assertEqual(len(orders),7)
        self.assertGreaterEqual(len(u),96)
        for family in n.SUPPORT_FAMILIES:
            self.assertEqual(len(orders[family]),769)
            self.assertEqual(len(set(orders[family])),769)
            for z in n.STRATA:
                s=n.forced_support(orders[family],u,769,z)
                self.assertEqual(len(s),96)
                self.assertEqual(len(s&u),24*z)
                self.assertEqual(len(s-u),96-24*z)
                for restart in range(8):
                    attacked=n.blind_attack(
                        row["full_public_input_sha256"],family,z,s,u,restart)
                    self.assertEqual(len(set(attacked)),4)
                    self.assertTrue(set(attacked)<=s)
                    self.assertEqual(len(set(attacked)&u),z)

    def test_exact_two_block_hypergeometric_number_of_missed_attacks(self):
        for sizea in range(0,7):
            for sizeb in range(0,7):
                for q in range(0,min(4,sizea)+1):
                    for r_other in range(0,min(4-q,sizeb)+1):
                        for t_a in range(sizea+1):
                            for t_b in range(sizeb+1):
                                inside=set(range(t_a))
                                outside=set(range(t_b))
                                total=miss=0
                                for s1 in combinations(range(sizea),q):
                                    for s2 in combinations(range(sizeb),r_other):
                                        total+=1
                                        if not (set(s1)&inside or set(s2)&outside):
                                            miss+=1
                                self.assertEqual(
                                    total,comb(sizea,q)*comb(sizeb,r_other))
                                self.assertEqual(
                                    miss,comb(sizea-t_a,q)*comb(sizeb-t_b,r_other))
                                self.assertEqual(
                                    Fraction(miss,total),
                                    Fraction(
                                        comb(sizea-t_a,q)*comb(sizeb-t_b,r_other),
                                        comb(sizea,q)*comb(sizeb,r_other)))

    def test_registered_cached_conditional_probabilities_exact(self):
        cached=n.conditional_prob_lookup()
        for z in n.STRATA:
            a=24*z;b=96-a
            for ti in range(min(a,16)+1):
                for to in range(min(b,16)+1):
                    expected=1-Fraction(
                        comb(a-ti,z)*comb(b-to,4-z),
                        comb(a,z)*comb(b,4-z))
                    self.assertAlmostEqual(cached[(z,ti,to)],float(expected),places=14)
        self.assertEqual(cached[(0,0,0)],0)
        self.assertEqual(cached[(4,0,0)],0)

    def test_original_equal_budget_auditors_and_ordered_policy_draw(self):
        seen=set(n.past_source_digests())
        row=n.generate_source(40,512,n.FAMILIES[0],seen)
        p=n.public_proposal(row)
        old=old24.supports(p["pred"],p["header"])
        plans=n.audit_plans(old,row["full_public_input_sha256"])
        self.assertEqual(set(plans),set(n.ARMS))
        for arm in n.ARMS:
            self.assertEqual(len(plans[arm]),512)
            self.assertTrue(all(len(set(s))==16 for s in plans[arm]))
        draws=[n.mixed_choice(row["full_public_input_sha256"],i) for i in range(128)]
        self.assertTrue(set(draws)<=set(n.ARMS))
        self.assertGreater(len(set(draws)),2)

    def test_ideal_source_conditional_blind_floor(self):
        for m in (769,1537):
            val=sum(Fraction(w,16)*n.ideal_floor(m,a)
                    for a,w in zip(n.ARMS,n.WEIGHTS))
            self.assertGreater(val,0)
            self.assertLess(val,n.ideal_floor(m,"uniform"))


if __name__=="__main__":
    unittest.main()
