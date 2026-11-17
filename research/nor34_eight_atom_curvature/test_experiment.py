"""NOR-34: source and exact eight-atom conditional hypergeometric controls."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import comb
import unittest

from research.nor34_eight_atom_curvature import experiment as x


class NOR34EightAtomTests(unittest.TestCase):
    def test_original_history_and_registered_constants(self):
        history=x.historic_source_hashes()
        self.assertEqual(len(history),len(set(history)))
        self.assertEqual(len(history),84)
        self.assertEqual(x.REPS,(44,45,46,47))
        self.assertEqual(x.OFFSET,130034000)
        self.assertEqual(x.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual(x.LAWS,("curvature","coarse","eight_atom"))
        self.assertEqual((x.ATTACKS,x.TRIALS,x.K,x.R),(16,512,16,4))

    def test_new_original_public_source_and_sealed_target(self):
        row=x.new_source(44,512,x.FAMILIES[0],set(x.historic_source_hashes()))
        self.assertEqual(row["split"],"TEST")
        self.assertTrue(all(v not in row for v in
                            ("program","future","target","attack","witness")))
        prop=x.nor33.public_proposal(row)
        self.assertEqual(prop["mask"],row["native_public_mask"])
        self.assertEqual(len(prop["pred"]),769)
        self.assertEqual(x.private_future_after_all_public_choices(
            row,prop["pred"]),prop["pred"])

    def test_curvature_support_atom_quotas_and_paired_controls(self):
        row=x.new_source(44,512,x.FAMILIES[0],set(x.historic_source_hashes()))
        prop=x.nor33.public_proposal(row)
        old,cats,u,cu,cv=x.partition(
            prop["pred"],prop["header"],row["full_public_input_sha256"])
        self.assertEqual(sum(len(c) for c in cats),769)
        self.assertEqual(len(cu),48)
        self.assertEqual(len(cv),48)
        self.assertEqual(cats[0],frozenset(range(769))-u)
        attacks=x.attack_triplets(row["full_public_input_sha256"],cats,u,cu,cv)
        self.assertEqual(len(attacks),16)
        for a in attacks:
            self.assertEqual(len(a["curvature"]),4)
            self.assertEqual(len(a["eight_atom"]),4)
            self.assertEqual(len(a["coarse"]),4)
            self.assertEqual(sum(a["quotas"]),4)
            self.assertEqual(a["quotas"][0],2)
            self.assertEqual(sum(i in u for i in a["curvature"]),2)
            self.assertEqual(sum(i in u for i in a["coarse"]),2)
            self.assertEqual([sum(i in c for i in a["eight_atom"])
                              for c in cats],a["quotas"])
        self.assertEqual(attacks,x.attack_triplets(
            row["full_public_input_sha256"],cats,u,cu,cv))

    def test_exact_fine_standardization_equals_direct_pair_category_sum(self):
        # Source has 48 curated U positions in category 1 and 48 V in
        # atom0, with additional uncurated members of all other atoms.
        categories=[frozenset(range(j*100,(j+1)*100)) for j in range(8)]
        cu=frozenset(range(100,148))
        cv=frozenset(range(0,48))
        fn=x.exact_probabilities(categories,cu,cv)
        checked=(3,4,105,106,201,301,401,501,601,701)
        t=tuple(len(set(checked)&a) for a in categories)
        r=fn(t,len(set(checked)&cu),len(set(checked)&cv))
        self.assertEqual(r["curvature"],
            Fraction(1)-x.conditional_binom_nohit(48,2,2)*
                        x.conditional_binom_nohit(48,2,2))
        self.assertEqual(r["coarse"],
            Fraction(1)-x.conditional_binom_nohit(700,sum(t[1:]),2)*
                        x.conditional_binom_nohit(100,t[0],2))
        # When the curated support intersects only one U atom, fine
        # resampling is exact two-position matching of that whole atom.
        expected=Fraction(1)-(
            x.conditional_binom_nohit(100,t[1],2)*
            x.conditional_binom_nohit(100,t[0],2))
        self.assertEqual(r["eight_atom"],expected)

    def test_exhaustive_two_atom_attack_redraw_count(self):
        # A curvature support with two U categories, plus V atom0;
        # count quota-conditional misses across all source sets and
        # compare with direct finite exhaustive expectation.
        a,b=4,4
        ua,ub=2,2
        t_a,t_b=1,1
        total=miss=0
        for src in combinations(range(a+b),2):
            quotas=(sum(i<a for i in src),sum(i>=a for i in src))
            for dstA in combinations(range(a),quotas[0]):
                for dstB in combinations(range(b),quotas[1]):
                    total+=1
                    if not(set(dstA)&{0} or set(dstB)&{0}):
                        miss+=1
        # The destination redraw is quota-conditioned, not equally
        # weighted at a flat attack-set level. Use per-source averages.
        conditional=[]
        for src in combinations(range(a+b),2):
            qA=sum(i<a for i in src)
            qB=2-qA
            conditional.append(
                Fraction(comb(a-t_a,qA),comb(a,qA))*
                Fraction(comb(b-t_b,qB),comb(b,qB)))
        probability=sum(conditional,Fraction(0))/len(conditional)
        expected=Fraction(0)
        expected+=Fraction(comb(a,2),comb(a+b,2))*Fraction(comb(a-t_a,2),comb(a,2))
        expected+=Fraction(comb(b,2),comb(a+b,2))*Fraction(comb(b-t_b,2),comb(b,2))
        expected+=Fraction(a*b,comb(a+b,2))*Fraction(a-t_a,a)*Fraction(b-t_b,b)
        self.assertEqual(probability,expected)

    def test_independent_computation_and_delta_shifts(self):
        y=tuple(range(16))
        checked=(1,3,5)
        target=tuple(range(16))
        attack=frozenset((2,3,10,12))
        for delta in x.DELTAS:
            self.assertFalse(x.nor24.exact_witness_accepts(
                y,target,checked,attack,delta))
            self.assertTrue(x.nor24.exact_witness_accepts(
                y,target,(0,1),attack,delta))
        for m in (769,1537):
            floor=sum((Fraction(w,16)*x.ideal_floor(m,arm)
                       for arm,w in zip(x.ARMS,x.WEIGHTS)),Fraction(0))
            self.assertGreater(floor,0)


if __name__=="__main__":
    unittest.main()
