"""NOR-32 registered nested-audit coverage, new source and commitment tests."""
from __future__ import annotations
from fractions import Fraction
from hashlib import sha256
from math import comb
import random
import unittest

from research.nor32_nested_audit_depth import source as s
from research.nor32_nested_audit_depth import sampler as p
from research.nor32_nested_audit_depth import attacks as a
from research.nor32_nested_audit_depth import grade as g
from research.nor31_matched_root import source as nor31
from research.nor30_minimax_audits import experiment as nor30
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor27_seed_leakage import experiment as n27
from research.nor28_commit_entropy import experiment as n28
from research.nor29_early_entropy import prepare as n29
from research.nor21_degree_scaling import experiment as nor21


class NOR32RegisteredTests(unittest.TestCase):
    def test_registered_grid_new_source_separation(self):
        self.assertEqual(s.OFFSET,160032000)
        self.assertEqual(s.REPS,(26,27,28,29))
        self.assertEqual(s.N_SOURCES,24)
        self.assertEqual(s.REPLICAS,32)
        self.assertEqual(p.TRIALS,2048)
        self.assertEqual(p.DEPTHS,(512,1024,1536,2048))
        self.assertEqual(g.N_DECISIONS,23592960)
        self.assertEqual(a.TOTAL,9216)
        self.assertEqual((p.T,p.K,a.ATTACK_RADIUS),(8,16,4))
        self.assertEqual(s.PRIOR_NUMERATORS,(295,246,153,78))
        previous={n24.source_seed(rep,d,f) for rep in n24.TRAIN_REPS+n24.TEST_REPS
                  for d in n24.DEGREES for f in n24.FAMILIES}
        for mod,reps in ((n25,n25.REPLICATES),(n26,n26.REPS),(n27,n27.REPS),
                         (n28,n28.REPS),(n29,n29.REPS),(nor31,nor31.REPS)):
            previous|={mod.source_seed(rep,d,f) for rep in reps
                       for d in (512,1024) for f in mod.FAMILIES}
        previous|={nor30.source_seed(rep,d,f) for rep in nor30.REPLICATES
                   for d in nor30.DEGREES for f in nor30.FAMILIES}
        fresh={s.source_seed(rep,d,f) for rep in s.REPS
               for d in s.DEGREES for f in s.FAMILIES}
        self.assertEqual(len(fresh),24)
        self.assertTrue(previous.isdisjoint(fresh))

    def test_native_public_mask_parity(self):
        row=s.generate_source(26,512,s.FAMILIES[0],set())
        self.assertEqual(row["split"],"TEST")
        self.assertTrue(all(k not in row for k in
                            ("program","future","nonce","attack","root","witness")))
        public,head=n24.public_only_case(row)
        self.assertIsNone(public.program)
        self.assertEqual(public.future,())
        original=nor21.native_mask(public,head,512)
        self.assertEqual(original,row["original_public_accept_mask"])
        proposal=n24.public_proposal(row)
        self.assertEqual(original,proposal["mask"])
        self.assertEqual(len(proposal["predicted"]),769)
        self.assertEqual(len(p.risk_order(proposal["predicted"],head,
                                          row["full_public_input_sha256"])),64)

    def test_independent_roots_and_nested_sampler(self):
        root=bytes(range(32))
        other=bytes(range(1,33))
        self.assertNotEqual(p.hash_root(root),p.hash_root(other))
        self.assertNotEqual(p.subkey(root,"calibration"),
                            p.subkey(root,"evaluation"))
        self.assertNotEqual(p.subkey(root,"evaluation"),
                            p.subkey(other,"evaluation"))
        order=tuple(range(64))
        digest="c"*64
        for arm in p.ARMS:
            for phase in p.PHASES:
                plans=[p.draw(root,phase,digest,arm,j,769,order)
                       for j in range(32)]
                self.assertTrue(all(len(q)==len(set(q))==16 for q in plans))
                if arm=="hybrid_50":
                    self.assertTrue(all(q[:8]==order[:8] and
                                        set(q[8:]).isdisjoint(order[:8])
                                        for q in plans))
                if arm=="static_prior":
                    self.assertTrue(all(q==order[:16] for q in plans))

    def test_greedy_attacks_matched_budget_and_commitments(self):
        m=128
        digest="a"*64
        plans=[tuple((i+5*j)%m for i in range(16)) for j in range(36)]
        cv=a.exposure_masks(plans,m)
        for arm in p.ARMS:
            for rep in range(4):
                changed,hits=a.greedy(cv,digest,arm,rep)
                self.assertEqual(len(set(changed)),4)
                self.assertEqual(hits,sum(bool(set(changed)&set(plan)) for plan in plans))
                self.assertEqual((changed,hits),a.greedy(cv,digest,arm,rep))
        opening={"source_sha256":digest,"auditor":"hybrid_50","depth":512,
                 "attack_class":"independent_calibration","replica":0,
                 "changed_indices":[2,6,8,13],"training_covered_schedules":7,
                 "nonce_hex":"f"*64}
        prior=a.commitment(opening)
        self.assertEqual(prior,a.commitment(dict(opening)))
        for field,val in (
            ("depth",1024),("replica",1),("changed_indices",[2,6,8,14]),
            ("nonce_hex","0"*64),("attack_class","leaked_evaluation_root")):
            change=dict(opening)
            change[field]=val
            self.assertNotEqual(prior,a.commitment(change))

    def test_prefix_coverage_monotone_and_zero_evasion(self):
        plans=[(0,1),(2,3),(4,5),(6,7)]
        for depth in range(1,len(plans)):
            earlier=set(i for p in plans[:depth] for i in p)
            later=set(i for p in plans[:depth+1] for i in p)
            self.assertTrue(earlier.issubset(later))
            self.assertGreaterEqual(10-len(earlier),10-len(later))
        assert len(set(range(10)).difference(set(plans[0])))>=4
        assert len(set(range(10)).difference(set(i for p in plans for i in p)))<4

    def test_ideal_uniform_expected_unseen_and_floor(self):
        for m in (769,1537):
            floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
            self.assertEqual(floor,n25.blind_floor(m,16,8,4))
            expected=lambda n:(m-8)*(1-8/(m-8))**n
            self.assertGreater(expected(512),expected(1024))
            self.assertGreater(expected(1024),expected(1536))
            self.assertGreater(expected(1536),expected(2048))
            self.assertLess(expected(2048),4)
        self.assertGreater((1537-8)*(1-8/(1537-8))**512,80)


if __name__=="__main__":
    unittest.main()
