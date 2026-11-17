"""NOR-36 pre-data deterministic source, greedy game and exact witness tests."""
from __future__ import annotations
from itertools import combinations
from math import comb
from fractions import Fraction
import unittest

from research.nor36_policyaware_curvature import experiment as n


class NOR36PreDataTests(unittest.TestCase):
    def test_historical_original_receipts_and_fixed_new_roster(self):
        roster=n.history()
        self.assertEqual(len(roster),len(set(roster)))
        self.assertEqual(len(roster),132)
        self.assertEqual(n.REPS,(52,53,54,55))
        self.assertEqual(n.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual(len(n.POLICIES),6)
        self.assertEqual((n.CAL,n.EVAL,n.RESTARTS,n.K,n.R),(256,512,8,16,4))

    def test_new_source_public_proposal_precedes_original_private_truth(self):
        previous=set(n.history())
        row=n.generate_source(52,512,n.FAMILIES[0],previous)
        for name in ("program","future","target","witness","attack"):
            self.assertNotIn(name,row)
        self.assertEqual(row["split"],"TEST")
        prop=n.proposal(row)
        self.assertEqual(prop["mask"],row["original_native_public_mask"])
        self.assertEqual(len(prop["pred"]),769)
        self.assertEqual(n.sealed_truth(row,prop["pred"]),prop["pred"])

    def test_equal_atom_budgets_and_new_phase_separation(self):
        seen=set(n.history())
        row=n.generate_source(52,512,n.FAMILIES[0],seen)
        p=n.proposal(row)
        old,cats,u,cu,cv,bu,bv,scores=n.public_regions(
            p["pred"],p["header"],row["full_public_input_sha256"])
        src=row["full_public_input_sha256"]
        self.assertNotEqual(n.phase_tag(src,"cal"),n.phase_tag(src,"eval"))
        plans_cal=n.schedules(old,cats,scores,src,"cal")
        plans_eval=n.schedules(old,cats,scores,src,"eval")
        self.assertEqual(len(plans_cal["original"]["hybrid_50"]),256)
        self.assertEqual(len(plans_eval["original"]["hybrid_50"]),512)
        self.assertNotEqual(plans_cal["original"]["hybrid_50"][:4],
                            plans_eval["original"]["hybrid_50"][:4])
        for arm in n.ARMS:
            for t in range(5):
                ref=set(plans_eval["original"][arm][t])
                original_counts=[len(ref&atom) for atom in cats]
                for mode in plans_eval:
                    chosen=plans_eval[mode][arm][t]
                    self.assertEqual(len(chosen),len(set(chosen)))
                    self.assertEqual(len(chosen),16)
                    self.assertEqual([len(set(chosen)&a) for a in cats],original_counts)

    def test_policy_defender_weights_and_all_coordinate_greedy_attack(self):
        for arm in n.POLICIES:
            self.assertEqual(sum(w for _,_,w in n.pure_components(arm)),16)
        m=24
        cov={(mode,arm):[0]*m
             for mode in ("original","atom_random","atom_curvature50")
             for arm in n.ARMS}
        for mode,arm in cov:
            for i in range(m):
                cov[(mode,arm)][i]=1<<((i+len(mode)+len(arm))%19)
        for arm in n.POLICIES:
            sets=[n.greedy_attack(cov,arm,"a"*64,rep) for rep in range(8)]
            self.assertTrue(all(len(set(x))==4 for x in sets))
            self.assertTrue(all(set(x)<=set(range(m)) for x in sets))
            self.assertEqual(sets,[n.greedy_attack(cov,arm,"a"*64,rep)
                                   for rep in range(8)])

    def test_blind_support_controls_and_both_integer_shifts(self):
        row=n.generate_source(52,512,n.FAMILIES[0],set(n.history()))
        p=n.proposal(row)
        old,cats,u,cu,cv,bu,bv,scores=n.public_regions(
            p["pred"],p["header"],row["full_public_input_sha256"])
        attacks=n.blind_attacks(row["full_public_input_sha256"],
                                 cats,u,cu,cv,bu,bv,len(p["pred"]))
        self.assertEqual(set(attacks),set(n.CONTROLS))
        for attack_group in attacks.values():
            self.assertEqual(len(attack_group),8)
            self.assertTrue(all(len(set(a))==4 for a in attack_group))
        for a in attacks["top_curvature"]:
            self.assertEqual(len(set(a)&u),2)
        for a,b in zip(attacks["top_curvature"],attacks["eight_atom_matched"]):
            self.assertEqual([sum(x in c for x in a) for c in cats],
                             [sum(x in c for x in b) for c in cats])
        predicted=tuple(range(12))
        for d in n.DELTAS:
            self.assertFalse(n.n24.exact_witness_accepts(
                predicted,predicted,(1,3,5),frozenset((3,4,6,7)),d))
            self.assertTrue(n.n24.exact_witness_accepts(
                predicted,predicted,(0,1,2),frozenset((3,4,6,7)),d))

    def test_uniform_attack_duality_and_positive_blind_floor(self):
        for m in range(5,9):
            for k in range(1,4):
                for r in range(1,4):
                    expected=1-Fraction(comb(m-k,r),comb(m,r))
                    for inspected in combinations(range(m),k):
                        actual=sum(bool(set(inspected)&set(changed))
                                   for changed in combinations(range(m),r))
                        self.assertEqual(Fraction(actual,comb(m,r)),expected)
        for m in (769,1537):
            f=sum((Fraction(w,16)*n.n35.nor34.ideal_floor(m,arm)
                   for arm,w in zip(n.ARMS,n.WEIGHTS)),Fraction(0))
            self.assertGreater(f,0)
            self.assertLess(f,1)

    def test_direct_expected_top_vs_sampled_source_attack_world(self):
        row=n.generate_source(52,512,n.FAMILIES[0],set(n.history()))
        p=n.proposal(row)
        old,cats,u,cu,cv,bu,bv,scores=n.public_regions(
            p["pred"],p["header"],row["full_public_input_sha256"])
        plans=n.schedules(old,cats,scores,row["full_public_input_sha256"],"eval")
        t=n.exact_policy_top(plans,cu,cv)
        self.assertEqual(set(t),set(n.POLICIES))
        self.assertTrue(all(0<=x<=1 for x in t.values()))
        self.assertGreater(t["atom_curv_h50"],t["original_h50"])
        self.assertGreater(t["atom_curv_mix"],t["original_mix"])


if __name__=="__main__":
    unittest.main()
