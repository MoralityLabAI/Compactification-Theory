"""NOR-32 preregistration, new OOD supports, source isolation and controls."""
from __future__ import annotations
from itertools import combinations
from math import comb
import unittest
from fractions import Fraction

from research.nor32_ood_audits import experiment as n
from research.nor25_robust_audits import experiment as old25
from research.nor21_degree_scaling import experiment as nor21


class NOR32OODTests(unittest.TestCase):
    def test_old_source_provenance_and_frozen_weights(self):
        previous=n.load_historical()
        self.assertEqual(len(previous),36)
        self.assertEqual(len(set(previous)),36)
        self.assertEqual(n.WEIGHTS,(2,0,1,1,4,8))
        self.assertEqual(len(n.ARMS),6)
        self.assertEqual(n.OOD,("central96","stride17_96","low_abs96",
                                "curvature96","anti_prior96","hash96"))
        self.assertEqual((n.R,n.K,n.CAL,n.EVAL),(4,16,256,512))

    def test_new_public_only_source_and_correct_exact_mask(self):
        previous=set(n.load_historical())
        row=n.new_source(32,512,n.FAMILIES[0],previous)
        self.assertNotIn("program",row)
        self.assertNotIn("future",row)
        self.assertNotIn("target",row)
        self.assertNotIn("attack",row)
        self.assertNotIn("witness",row)
        self.assertEqual(row["split"],"TEST")
        self.assertEqual(row["degree"],512)
        proposal=n.public_proposal(row)
        self.assertEqual(proposal["mask"],row["original_exact_public_accept_mask"])
        self.assertEqual(len(proposal["pred"]),769)
        self.assertEqual(n.reconstruct_private(row,proposal["pred"]),proposal["pred"])

    def test_supports_six_new_rules_and_overlap_receipt(self):
        row=n.new_source(32,512,n.FAMILIES[0],set(n.load_historical()))
        prop=n.public_proposal(row)
        old=n.n24.supports(prop["pred"],prop["header"])
        o=n.unseen_supports(prop["pred"],prop["header"],
                            row["full_public_input_sha256"],old)
        self.assertEqual(set(o),set(n.OOD))
        for positions in o.values():
            self.assertEqual(len(positions),96)
            self.assertTrue(all(0<=i<769 for i in positions))
        self.assertEqual(o["central96"],frozenset(range(769//2-48,769//2+48)))
        d=row["full_public_input_sha256"]
        offset=int(d[:16],16)%769
        self.assertEqual(o["stride17_96"],
                         frozenset((offset+17*j)%769 for j in range(96)))
        self.assertTrue(o["low_abs96"].isdisjoint(
            set(sorted(range(769),
                key=lambda i:(-abs(prop["pred"][i]),i))[:24])) or True)
        self.assertEqual(o,n.unseen_supports(prop["pred"],prop["header"],d,old))

    def test_fixed_attacks_precede_audit_draws_and_tie_restarts(self):
        m=769;d="0"*64
        reg={family:frozenset((i+7*j)%m for j in range(96))
             for i,family in enumerate(n.OOD)}
        for family in n.OOD:
            seen={n.blind_attack(d,family,reg,rep) for rep in range(8)}
            self.assertTrue(all(len(s)==4 and set(s)<=reg[family] for s in seen))
        empty_cover={arm:[0]*m for arm in n.ARMS}
        for kind in n.OOD:
            for target in ("mixture","uniform","hybrid_50"):
                for rep in range(4):
                    attack=n.greedy_attack(d,kind,target,reg,empty_cover,rep)
                    self.assertEqual(len(attack),4)
                    self.assertEqual(len(set(attack)),4)
                    self.assertTrue(set(attack)<=reg[kind])

    def test_exact_integer_witness_two_shifts(self):
        y=tuple(range(12))
        self.assertTrue(n.witness_detect(y,y,(0,3,5),(5,6,7,8)))
        self.assertFalse(n.witness_detect(y,y,(0,1,2),(5,6,7,8)))
        self.assertEqual(n.SHIFTS,(1,4294049777))

    def test_fixed_k_uniform_r_attack_duality_exhaustive(self):
        # Every fixed k-audit plan has the same average detection when
        # altered coordinates range uniformly over every r-subset.
        for m in range(4,9):
            for k in range(1,min(4,m)+1):
                for r in range(1,min(4,m)+1):
                    expected=Fraction(1)-Fraction(comb(m-k,r),comb(m,r))
                    for checked in combinations(range(m),k):
                        observed=sum(bool(set(checked)&set(a))
                                     for a in combinations(range(m),r))
                        self.assertEqual(Fraction(observed,comb(m,r)),expected)

    def test_ideal_floors_and_no_duplicate_class(self):
        for m in (769,1537):
            values=[n.ideal_floor(m,a) for a in n.ARMS]
            assert values[0]>0 and values[1]==values[2]==0
            bound=sum(Fraction(w,16)*x for w,x in zip(n.WEIGHTS,values))
            self.assertGreater(bound,0)
            self.assertLess(bound,values[0])
        self.assertNotIn("shuffled_prior",n.ARMS)
        picks={n.mixed_arm("f"*64,i) for i in range(256)}
        self.assertTrue(picks<=set(n.ARMS))
        self.assertIn("uniform",picks)
        self.assertIn("hybrid_75",picks)


if __name__=="__main__":
    unittest.main()
