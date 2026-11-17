"""NOR-27 frozen leakage protocol, exact conditional floor and native source parity."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
import unittest

from research.nor27_seed_leakage import experiment as e
from research.nor24_adaptive_audits import experiment as n24
from research.nor25_robust_audits import experiment as n25
from research.nor26_policy_aware_audits import experiment as n26
from research.nor21_degree_scaling import experiment as nor21


class NOR27LeakageTests(unittest.TestCase):
    def test_registered_parameters_and_new_source_isolation(self):
        self.assertEqual(e.REPS,(8,9))
        self.assertEqual(e.OFFSET,119327000)
        self.assertEqual(e.LEAK_REVEAL,(0,2,4,6,8))
        self.assertEqual(e.SEED_LEAK_BITS,(0,2,4))
        self.assertEqual(e.PREVIEW_TRIALS,1024)
        self.assertEqual(e.CAL_PER_BUCKET,128)
        self.assertEqual(e.EVAL_PER_BUCKET,256)
        self.assertEqual(e.ATTACK_REPLICATES,4)
        self.assertEqual(e.PRIOR_NUMERATORS,(295,246,153,78))
        self.assertEqual((e.TARGETED,e.RESERVE,e.RADIUS),(8,8,4))
        previous={n24.source_seed(r,d,f) for r in n24.TRAIN_REPS+n24.TEST_REPS
                  for d in n24.DEGREES for f in n24.FAMILIES}
        previous|={n25.source_seed(r,d,f) for r in n25.REPLICATES
                   for d in n25.DEGREES for f in n25.FAMILIES}
        previous|={n26.source_seed(r,d,f) for r in n26.REPS
                   for d in n26.DEGREES for f in n26.FAMILIES}
        fresh={e.source_seed(r,d,f) for r in e.REPS
               for d in e.DEGREES for f in e.FAMILIES}
        self.assertTrue(previous.isdisjoint(fresh))
        self.assertEqual(len(fresh),12)

    def test_exact_conditional_hypergeometric_small_universes(self):
        for m in range(10,14):
            t=2
            u=3
            r=2
            universe=tuple(range(t,m))
            for ell in range(u+1):
                revealed=tuple(universe[:ell])
                allowed=tuple(i for i in universe if i not in revealed)
                q=Fraction(1)-Fraction(
                    comb(m-t-ell-r,u-ell),comb(m-t-ell,u-ell))
                lower=Fraction(1)
                for changed in combinations(allowed,r):
                    hits=sum(bool(set(changed)&set(s))
                             for s in combinations(allowed,u-ell))
                    rate=Fraction(hits,comb(len(allowed),u-ell))
                    self.assertEqual(rate,q)
                    lower=min(lower,rate)
                self.assertEqual(lower,q)
        self.assertEqual(e.preview_floor(769,8),0)
        self.assertEqual(e.preview_floor(1537,8),0)
        self.assertGreater(e.preview_floor(769,0),e.preview_floor(769,4))
        self.assertGreater(e.preview_floor(769,4),0)
        self.assertEqual(e.preview_floor(769,0),n25.blind_floor(769,16,8,4))

    def test_preview_attack_observes_only_prefix(self):
        m=769
        T=(3,4,6,7,8,9,10,11)
        secret=(23,41,100,150,300,400,500,600)
        for ell in e.LEAK_REVEAL:
            chosen=e.preview_attacker("f"*64,secret[:ell],T,m,ell)
            self.assertEqual(len(set(chosen)),4)
            self.assertTrue(set(chosen).isdisjoint(set(T)|set(secret[:ell])))
            self.assertEqual(chosen,e.preview_attacker("f"*64,secret[:ell],T,m,ell))
            # A second world with the same disclosed prefix yields the
            # same attacker decision, regardless of the hidden completion.
            self.assertEqual(chosen,e.preview_attacker("f"*64,
                                                       secret[:ell],T,m,ell))
            if ell==8:
                self.assertTrue(set(chosen).isdisjoint(set(secret)))
        with self.assertRaises(ValueError):
            e.preview_attacker("f"*64,secret[:2],T,m,4)

    def test_conditional_raw_seed_highbits_and_phase_isolation(self):
        d="d"*64
        for bits in e.SEED_LEAK_BITS:
            for bucket in range(1<<bits):
                cal={e.conditional_seed(d,bits,bucket,"cal",j) for j in range(32)}
                test={e.conditional_seed(d,bits,bucket,"eval",j) for j in range(32)}
                self.assertEqual(len(cal),32)
                self.assertEqual(len(test),32)
                self.assertTrue(cal.isdisjoint(test))
                self.assertTrue(all(s>>(64-bits)==bucket for s in cal|test))
        with self.assertRaises(ValueError):
            e.conditional_seed(d,4,16,"cal",0)

    def test_hash_expanded_sampler_and_training_attacker(self):
        m=769
        digest="a"*64
        T=tuple(range(8))
        for arm in e.SEED_AUDITORS:
            plans=[e.draw_schedule(digest,s,m,arm,T) for s in range(32)]
            self.assertTrue(all(len(x)==len(set(x))==16 for x in plans))
            if arm=="hybrid_50":
                self.assertTrue(all(p[:8]==T for p in plans))
                self.assertTrue(all(not (set(p[8:])&set(T)) for p in plans))
            exp=e.exposure_masks(plans,m)
            for attack_rep in range(e.ATTACK_REPLICATES):
                changed,count=e.greedy_calibration_attacker(
                    exp,digest,arm,2,1,attack_rep)
                self.assertEqual(len(set(changed)),4)
                self.assertEqual(count,
                    sum(bool(set(p)&set(changed)) for p in plans))
        self.assertEqual(
            e.draw_schedule(digest,21,m,"hybrid_50",T),
            e.draw_schedule(digest,21,m,"hybrid_50",T))

    def test_one_new_public_source_verifier_native_parity(self):
        row=e.new_source(8,512,e.FAMILIES[0],set())
        self.assertEqual(row["split"],"TEST")
        self.assertTrue(all(k not in row for k in
                            ("future","program","witness","audit_seed","attack")))
        case,header=n24.public_only_case(row)
        self.assertIsNone(case.program)
        self.assertEqual(case.future,())
        self.assertEqual(nor21.native_mask(case,header,512),
                         row["original_public_accept_mask"])
        prop=n24.public_proposal(row)
        regs=n24.supports(prop["predicted"],header)
        T=e.targeted_prefix(regs,row["full_public_input_sha256"])
        self.assertEqual(len(T),8)
        self.assertEqual(len(prop["predicted"]),769)


if __name__=="__main__":
    unittest.main()
