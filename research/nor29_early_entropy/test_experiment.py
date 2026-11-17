"""NOR-29 exact source isolation, salted attack binding and root-aware game tests."""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import comb
import unittest

from research.nor29_early_entropy import prepare as p
from research.nor29_early_entropy import challenge as c
from research.nor29_early_entropy import leaked_root_attacker as l
from research.nor29_early_entropy import grade as g
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor27_seed_leakage import experiment as n27
from research.nor28_commit_entropy import experiment as n28
from research.nor21_degree_scaling import experiment as nor21


class NOR29FirstProspectiveTests(unittest.TestCase):
    def test_source_and_grid_frozen(self):
        self.assertEqual(p.OFFSET,139329000)
        self.assertEqual(p.REPS,(12,13,14,15))
        self.assertEqual(p.N_SOURCES,24)
        self.assertEqual(p.REPLICAS,32)
        self.assertEqual(p.RADIUS,4)
        self.assertEqual(c.TRIALS,512)
        self.assertEqual(p.ARMS,("hybrid_50","uniform16","static_prior"))
        self.assertEqual(g.EXPECTED_DECISIONS,2359296)
        self.assertEqual(p.PRIOR_NUMERATORS,(295,246,153,78))
        prior={n24.source_seed(rep,d,f) for rep in n24.TRAIN_REPS+n24.TEST_REPS
               for d in n24.DEGREES for f in n24.FAMILIES}
        prior|={n25.source_seed(rep,d,f) for rep in n25.REPLICATES
                for d in n25.DEGREES for f in n25.FAMILIES}
        prior|={n26.source_seed(rep,d,f) for rep in n26.REPS
                for d in n26.DEGREES for f in n26.FAMILIES}
        prior|={n27.source_seed(rep,d,f) for rep in n27.REPS
                for d in n27.DEGREES for f in n27.FAMILIES}
        prior|={n28.source_seed(rep,d,f) for rep in n28.REPS
                for d in n28.DEGREES for f in n28.FAMILIES}
        fresh={p.source_seed(rep,d,f) for rep in p.REPS
               for d in p.DEGREES for f in p.FAMILIES}
        self.assertEqual(len(fresh),24)
        self.assertTrue(prior.isdisjoint(fresh))

    def test_original_commitment_mutation_and_attack_sampling(self):
        opening={"source_sha256":"a"*64,"attack_class":"blind_uniform",
                 "replica":0,"changed_indices":[0,2,3,7],"nonce_hex":"4"*64}
        expected=p.commit(opening)
        self.assertEqual(expected,p.commit(dict(opening)))
        self.assertEqual(len(expected),64)
        for key,value in (
            ("source_sha256","b"*64),("replica",1),
            ("changed_indices",[0,2,3,8]),("nonce_hex","5"*64),
            ("attack_class","early_root_aware"),
        ):
            modified=dict(opening)
            modified[key]=value
            self.assertNotEqual(expected,p.commit(modified))
        for rep in range(32):
            A=p.blind_set("f"*64,769,rep)
            self.assertEqual(len(set(A)),4)
            self.assertEqual(A,p.blind_set("f"*64,769,rep))

    def test_fresh_hmac_root_and_schedule_sampler(self):
        root=bytes(range(32))
        other=bytes(range(1,33))
        self.assertNotEqual(c.root_commitment(root),c.root_commitment(other))
        self.assertNotEqual(c.audit_key(root),c.audit_key(other))
        self.assertEqual(c.audit_key(root),c.audit_key(bytes(range(32))))
        m=769
        order=tuple(range(64))
        digest="e"*64
        for arm in p.ARMS:
            for trial in range(8):
                s=c.draw(root,digest,arm,trial,m,order)
                self.assertEqual(len(set(s)),16)
                self.assertTrue(all(0<=i<m for i in s))
                if arm=="hybrid_50":
                    self.assertEqual(s[:8],order[:8])
                    self.assertTrue(set(s[8:]).isdisjoint(order[:8]))
                if arm=="static_prior":
                    self.assertEqual(s,order[:16])
        self.assertNotEqual(c.draw(root,digest,"hybrid_50",0,m,order),
                            c.draw(root,digest,"hybrid_50",1,m,order))

    def test_policy_aware_greedy_uses_calibrated_batch(self):
        m=64
        plans=[]
        for j in range(32):
            plans.append(tuple((i+2*j)%m for i in range(16)))
        exposure=l.coverage(plans,m)
        self.assertEqual(len(exposure),m)
        for arm in p.ARMS:
            for rep in range(p.REPLICAS):
                attack,hits=l.greedy(exposure,"f"*64,arm,rep)
                self.assertEqual(len(attack),len(set(attack)))
                self.assertEqual(len(attack),4)
                self.assertEqual(hits,sum(bool(set(attack)&set(plan))
                                          for plan in plans))
                self.assertEqual((attack,hits),
                                 l.greedy(exposure,"f"*64,arm,rep))

    def test_finite_fixed_blind_floor_not_root_aware(self):
        for m in (769,1537):
            floor=Fraction(1)-Fraction(comb(m-12,8),comb(m-8,8))
            self.assertEqual(floor,n25.blind_floor(m,16,8,4))
            self.assertGreater(floor,0)
        # Every fixed index is hit in one of two roots. A seed-aware
        # response chooses its opposite and evades on both roots.
        audit={False:{0},True:{1}}
        fixed=[sum(int(i in audit[s]) for s in audit) for i in (0,1)]
        aware=sum(int((1 if s is False else 0) in audit[s]) for s in audit)
        self.assertEqual(fixed,[1,1])
        self.assertEqual(aware,0)

    def test_one_new_public_case_matches_original_native_verifier(self):
        row=p.generate_source(12,512,p.FAMILIES[0],set())
        case,h=n24.public_only_case(row)
        self.assertIsNone(case.program)
        self.assertEqual(case.future,())
        self.assertEqual(nor21.native_mask(case,h,512),
                         row["original_public_accept_mask"])
        pred=n24.public_proposal(row)
        self.assertEqual(pred["mask"],row["original_public_accept_mask"])
        self.assertEqual(len(pred["predicted"]),769)
        order=p.rank_order(pred["predicted"],h,
                            row["full_public_input_sha256"])
        self.assertEqual(len(order),64)


if __name__=="__main__":
    unittest.main()
