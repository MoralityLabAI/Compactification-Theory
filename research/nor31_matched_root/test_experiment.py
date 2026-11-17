"""NOR-31 frozen source split, matched greedy attack budget, two OS-root phases,
finite floor and salted commitment regressions, before first data.
"""
from __future__ import annotations
from fractions import Fraction
from hashlib import sha256
from math import comb
import random
import unittest

from research.nor31_matched_root import source as s
from research.nor31_matched_root import sampler as a
from research.nor31_matched_root import attacks as g
from research.nor31_matched_root import grade as d
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor27_seed_leakage import experiment as n27
from research.nor28_commit_entropy import experiment as n28
from research.nor29_early_entropy import prepare as n29
from research.nor30_minimax_audits import experiment as n30
from research.nor21_degree_scaling import experiment as nor21


class NOR31FrozenProspectiveTests(unittest.TestCase):
    def test_grid_and_new_sources(self):
        self.assertEqual(s.OFFSET,150031000)
        self.assertEqual(s.REPS,(22,23,24,25))
        self.assertEqual(s.N_SOURCES,24)
        self.assertEqual(s.REPLICAS,32)
        self.assertEqual(a.TRIALS,512)
        self.assertEqual(d.DECISIONS,2359296)
        self.assertEqual((a.T,a.K,g.ATTACK_RADIUS),(8,16,4))
        self.assertEqual(a.PHASES,("calibration","evaluation"))
        self.assertEqual(g.LABELS,("independent_calibration","leaked_evaluation_root"))
        prior={n24.source_seed(rep,d,f) for rep in n24.TRAIN_REPS+n24.TEST_REPS
               for d in n24.DEGREES for f in n24.FAMILIES}
        for mod,reps in ((n25,n25.REPLICATES),(n26,n26.REPS),
                         (n27,n27.REPS),(n28,n28.REPS),
                         (n29,n29.REPS)):
            prior|={mod.source_seed(rep,d,f) for rep in reps
                    for d in (512,1024) for f in mod.FAMILIES}
        prior|={n30.source_seed(rep,d,f) for rep in n30.REPLICATES
                for d in n30.DEGREES for f in n30.FAMILIES}
        fresh={s.source_seed(rep,d,f) for rep in s.REPS for d in s.DEGREES for f in s.FAMILIES}
        self.assertEqual(len(fresh),24)
        self.assertTrue(fresh.isdisjoint(prior))

    def test_native_verifier_and_public_future_on_new_source(self):
        row=s.generate_source(22,512,s.FAMILIES[0],set())
        self.assertEqual(row["split"],"TEST")
        self.assertFalse(any(x in row for x in ("program","future","root","attack","nonce","witness")))
        public,h=n24.public_only_case(row)
        self.assertIsNone(public.program)
        self.assertEqual(public.future,())
        mask=nor21.native_mask(public,h,512)
        self.assertEqual(mask,row["original_public_accept_mask"])
        p=n24.public_proposal(row)
        self.assertEqual(p["mask"],mask)
        self.assertEqual(len(p["predicted"]),769)
        self.assertEqual(len(a.risk_order(p["predicted"],p["header"],
                                          row["full_public_input_sha256"])),64)

    def test_phase_separation_and_unbiased_sampler(self):
        root1=bytes(range(32))
        root2=bytes(range(1,33))
        self.assertNotEqual(a.hash_root(root1),a.hash_root(root2))
        self.assertNotEqual(a.subkey(root1,"calibration"),a.subkey(root1,"evaluation"))
        self.assertNotEqual(a.subkey(root1,"calibration"),a.subkey(root2,"calibration"))
        self.assertEqual(len(a.subkey(root1,"calibration")),32)
        order=tuple(range(64))
        digest="f"*64
        for arm in a.ARMS:
            for trial in range(8):
                for phase in a.PHASES:
                    plan=a.draw(root1,phase,digest,arm,trial,769,order)
                    self.assertEqual(len(plan),len(set(plan)))
                    self.assertEqual(len(plan),16)
                    self.assertTrue(all(0<=j<769 for j in plan))
                    if arm=="hybrid_50":
                        self.assertEqual(plan[:8],order[:8])
                        self.assertFalse(set(plan[8:])&set(order[:8]))
        self.assertNotEqual(
            a.draw(root1,"calibration",digest,"hybrid_50",0,769,order),
            a.draw(root1,"evaluation",digest,"hybrid_50",0,769,order))

    def test_identical_attacker_search_budget_and_tie_seeds(self):
        m=769
        digest="a"*64
        plans1=[tuple((i+2*j)%m for i in range(16)) for j in range(64)]
        plans2=[tuple((i+13*j)%m for i in range(16)) for j in range(64)]
        mask1=g.exposure_masks(plans1,m)
        mask2=g.exposure_masks(plans2,m)
        for arm in a.ARMS:
            for replica in range(g.REPLICAS):
                pick,hit=g.greedy(mask1,digest,arm,replica)
                self.assertEqual((pick,hit),g.greedy(mask1,digest,arm,replica))
                self.assertEqual(len(set(pick)),4)
                self.assertEqual(hit,sum(bool(set(pick)&set(p)) for p in plans1))
                other,hit2=g.greedy(mask2,digest,arm,replica)
                self.assertEqual(hit2,sum(bool(set(other)&set(p)) for p in plans2))

    def test_cryptographic_commitment_binding_explicit_scope(self):
        x={
            "source_sha256":"a"*64,"auditor":"hybrid_50",
            "attack_class":"independent_calibration","replica":0,
            "changed_indices":[1,2,3,9],"training_covered_schedules":15,
            "nonce_hex":"4"*64}
        old=g.commitment(x)
        self.assertEqual(old,g.commitment(dict(x)))
        self.assertEqual(len(old),64)
        for key,val in (("source_sha256","b"*64),("auditor","uniform16"),
                        ("attack_class","leaked_evaluation_root"),("replica",1),
                        ("changed_indices",[1,2,3,10]),
                        ("training_covered_schedules",16),("nonce_hex","5"*64)):
            altered=dict(x)
            altered[key]=val
            self.assertNotEqual(g.commitment(altered),old)

    def test_fixed_blind_floor_not_applicable_to_root_aware_attack(self):
        for m in (769,1537):
            val=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
            self.assertEqual(val,n25.blind_floor(m,16,8,4))
        # An independent-root fixed attack is hit once across two roots.
        # A root-aware attack changes location and evades on both roots.
        checks={False:{0},True:{1}}
        fixed=[sum(int(i in checks[root]) for root in checks) for i in (0,1)]
        aware=sum(int((1 if root is False else 0) in checks[root])
                  for root in checks)
        self.assertEqual(fixed,[1,1])
        self.assertEqual(aware,0)


if __name__=="__main__":
    unittest.main()
