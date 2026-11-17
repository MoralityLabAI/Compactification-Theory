"""NOR-35 preregistered finite exact atom quotas, source isolation and witness tests."""
from __future__ import annotations

import unittest
from fractions import Fraction
from math import comb
from itertools import combinations

from research.nor35_intraatom_curvature import experiment as n


class NOR35Tests(unittest.TestCase):
    def test_source_and_historical_immutability(self):
        history=n.historical_ids()
        self.assertEqual(len(history),len(set(history)))
        self.assertEqual(len(history),108)
        self.assertEqual(n.REPS,(48,49,50,51))
        self.assertEqual(n.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual((n.K,n.R,n.TRIALS,n.ATTACK_REPS),(16,4,512,16))

    def test_new_original_public_source_and_native_target(self):
        seen=set(n.historical_ids())
        row=n.source_public(48,512,n.FAMILIES[0],seen)
        self.assertEqual(row["split"],"TEST")
        for name in ("program","future","target","attack","witness"):
            self.assertNotIn(name,row)
        proposal=n.proposal(row)
        self.assertEqual(proposal["mask"],row["original_native_public_mask"])
        self.assertEqual(len(proposal["pred"]),769)
        true=n.sealed_native_truth(row,proposal["pred"])
        self.assertEqual(true,proposal["pred"])

    def test_eight_atom_partition_and_blind_attack_membership(self):
        row=n.source_public(48,512,n.FAMILIES[0],set(n.historical_ids()))
        pub=n.proposal(row)
        old,cats,u,cu,cv,bu,bv,scores=n.source_atoms(
            pub["pred"],pub["header"],row["full_public_input_sha256"])
        self.assertEqual(len(cats),8)
        self.assertEqual(sum(len(a) for a in cats),769)
        self.assertEqual(len(cu),len(cv))
        self.assertEqual(len(bu),len(bv))
        self.assertEqual(len(cu),48)
        self.assertEqual(len(bu),48)
        self.assertEqual(len(scores),769)
        attacks,quotas=n.attacks_for_source(row["full_public_input_sha256"],
                                             cats,u,cu,cv,bu,bv,769)
        self.assertEqual(set(attacks),set(n.ATTACKS))
        self.assertTrue(all(len(v)==16 for v in attacks.values()))
        for i in range(16):
            a=attacks["top_curvature"][i]
            b=attacks["bottom_curvature"][i]
            f=attacks["eight_atom_matched"][i]
            self.assertEqual(len(a),4)
            self.assertEqual(sum(j in u for j in a),2)
            self.assertEqual(sum(j in u for j in b),2)
            self.assertEqual(quotas[i][0],2)
            self.assertEqual(tuple(sum(j in c for j in f) for c in cats),quotas[i])
        self.assertEqual((attacks,quotas),n.attacks_for_source(
            row["full_public_input_sha256"],cats,u,cu,cv,bu,bv,769))

    def test_transforms_preserve_original_per_atom_quotas(self):
        row=n.source_public(48,512,n.FAMILIES[0],set(n.historical_ids()))
        pub=n.proposal(row)
        old,cats,u,cu,cv,bu,bv,scores=n.source_atoms(
            pub["pred"],pub["header"],row["full_public_input_sha256"])
        p=n.nor25.policy_schedules(
            old,"test-original-policy-quota-v1",71)
        for arm in n.ARMS:
            original=tuple(p[(arm,16)])
            v=[len(set(original)&a) for a in cats]
            for mode in n.TRANSFORMS:
                planned=n.transformed_plan(
                    original,cats,scores,row["full_public_input_sha256"],
                    arm,71,mode)
                self.assertEqual(len(planned),len(set(planned)))
                self.assertEqual(len(planned),16)
                self.assertEqual(v,[len(set(planned)&a) for a in cats])

    def test_exact_conditional_quota_invariance(self):
        row=n.source_public(48,512,n.FAMILIES[0],set(n.historical_ids()))
        pub=n.proposal(row)
        old,cats,u,cu,cv,bu,bv,scores=n.source_atoms(
            pub["pred"],pub["header"],row["full_public_input_sha256"])
        fn=n.nor34.exact_probabilities(cats,cu,cv)
        original=n.nor25.policy_schedules(old,"test-exact-prob-nor35",41)
        for arm in n.ARMS:
            c=tuple(original[(arm,16)])
            baseline=n.exact_analytic(fn,cats,cu,cv,bu,bv,c,769)
            for mode in n.TRANSFORMS:
                transform=n.transformed_plan(
                    c,cats,scores,row["full_public_input_sha256"],arm,41,mode)
                expected=n.exact_analytic(fn,cats,cu,cv,bu,bv,transform,769)
                self.assertEqual(expected["eight_atom_matched"],
                                 baseline["eight_atom_matched"])
                self.assertEqual(expected["full_uniform"],
                                 baseline["full_uniform"])

    def test_exact_two_block_and_full_uniform_duality(self):
        for m in range(5,10):
            for k in range(1,5):
                for r in range(1,4):
                    if k>m or r>m:continue
                    miss=Fraction(comb(m-k,r),comb(m,r))
                    for checked in combinations(range(m),k):
                        observed=sum(bool(set(checked)&set(a))
                                     for a in combinations(range(m),r))
                        self.assertEqual(Fraction(observed,comb(m,r)),1-miss)
        m=769
        top=set(range(48))
        bot=set(range(48,96))
        for inspect in ((),(0,1),(0,48),(0,1,48,49)):
            exact=n.two_block_detect(top,bot,inspect)
            self.assertEqual(exact,Fraction(1)-Fraction(
                comb(48-len(set(inspect)&top),2),comb(48,2))*Fraction(
                comb(48-len(set(inspect)&bot),2),comb(48,2)))

    def test_integer_witness_nonzero_shifts(self):
        y=tuple(range(16))
        plan=(0,4,8)
        attack=frozenset((2,4,10,12))
        for shift in n.DELTAS:
            self.assertFalse(n.nor24.exact_witness_accepts(
                y,y,plan,attack,shift))
            self.assertTrue(n.nor24.exact_witness_accepts(
                y,y,(0,1,3),attack,shift))


if __name__=="__main__":
    unittest.main()
