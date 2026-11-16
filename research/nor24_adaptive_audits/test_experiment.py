"""NOR-24 exact source isolation, bounded Bayesian update and blind controls."""
from __future__ import annotations

from itertools import combinations
from math import comb
import unittest

from research.nor24_adaptive_audits import experiment as n
from research.nor21_degree_scaling import experiment as nor21
from research.rrc_compactification import compound25 as corpus


class NOR24SequentialAuditTests(unittest.TestCase):
    def test_conditional_nohit_counts_exhaustive(self):
        for s in range(2, 9):
            universe = tuple(range(s))
            for r in range(s+1):
                for t in range(s+1):
                    remaining = set(range(t, s))
                    direct = sum(set(attack) <= remaining
                                 for attack in combinations(universe, r))
                    self.assertEqual(direct, n.conditional_nohit_count(s, t, r))
        self.assertEqual(n.conditional_nohit_count(96, 0, 4), comb(96, 4))

    def test_true_witness_rejects_exact_nonzero_shifts(self):
        pred = (12, -7, 19, 47, 5, 3)
        for shift in n.DELTAS:
            self.assertTrue(n.exact_witness_accepts(
                pred, pred, (0, 1), {4}, shift))
            self.assertFalse(n.exact_witness_accepts(
                pred, pred, (0, 4), {4}, shift))
            self.assertTrue(n.exact_witness_accepts(
                pred, pred, (), {4}, shift))
        self.assertFalse(n.exact_witness_accepts(
            pred, pred, (4,), {4}, n.DELTAS[1]))

    def test_preregistration_params_and_disjoint_seeds(self):
        self.assertEqual(n.OFFSET, 94324000)
        self.assertEqual(n.TRAIN_REPS, (0,1))
        self.assertEqual(n.TEST_REPS, (2,3))
        self.assertEqual(n.TRIALS, 1024)
        self.assertEqual(n.BUDGETS, (4,16,64))
        self.assertEqual(len(set(
            n.source_seed(rep,deg,f) for rep in range(4)
            for deg in n.DEGREES for f in n.FAMILIES)), 24)

    def test_support_sizes_schedules_and_privileged_label(self):
        pred = (1,) + tuple(i*i if i%7 == 0 else -i for i in range(1, 769))
        h = corpus.Header("I", 80, 32, 1)
        support = n.supports(pred, h)
        self.assertEqual({k:len(support[k]) for k in n.KINDS[:3]},
                         {k:96 for k in n.KINDS[:3]})
        self.assertGreater(len(support[n.STRESS]), 16)
        prior = dict(zip(n.KINDS, n.PRIOR_WEIGHTS))
        plans = n.policy_schedules(support, prior, "f"*64, 0)
        self.assertEqual(plans, n.policy_schedules(support, prior, "f"*64, 0))
        self.assertEqual(len(plans), len(n.ARMS)-1+len(n.ALL_KINDS))
        for arm in n.ARMS[:-1]:
            self.assertEqual(len(set(plans[arm])), 64)
            self.assertTrue(all(0<=i<769 for i in plans[arm]))
        for kind in n.ALL_KINDS:
            self.assertTrue(set(plans[("oracle_support",kind)])<=support[kind])
        self.assertEqual(len(set(plans["posterior"][:16])),16)

    def test_informed_evasion_for_each_frozen_policy(self):
        pred = (1,) + tuple(0 if i%13 else i for i in range(1, 769))
        h = corpus.Header("N", 96, 24, 1)
        regs = n.supports(pred, h)
        plans = n.policy_schedules(regs, dict(zip(n.KINDS,n.PRIOR_WEIGHTS)),
                                   "b"*64,2)
        for kind in n.ALL_KINDS:
            for arm in n.ARMS:
                plan = plans[arm] if arm != "oracle_support" else plans[(arm,kind)]
                for k in n.BUDGETS:
                    for r in n.RADII:
                        unchecked = n.informed_attack(plan[:k],regs[kind],r)
                        self.assertEqual(len(unchecked),r)
                        self.assertFalse(set(plan[:k])&unchecked)
                        self.assertTrue(n.exact_witness_accepts(
                            pred,pred,plan[:k],unchecked,1))

    def test_small_real_public_source_only_and_native_mask(self):
        source = n.generate_source(2,512,n.FAMILIES[0],set())
        self.assertEqual(source["split"],"TEST")
        self.assertNotIn("program",source)
        self.assertNotIn("future",source)
        self.assertNotIn("attack",source)
        c,h=n.public_only_case(source)
        self.assertIsNone(c.program)
        self.assertEqual(c.future,())
        proposal=n.public_proposal(source)
        self.assertEqual(proposal["mask"],source["original_public_accept_mask"])
        self.assertEqual(nor21.native_mask(c,h,512),proposal["mask"])
        self.assertEqual(n.private_truth_only_after_proposal(
            source,proposal["predicted"]), proposal["predicted"])
        for kind,r in ((k,16) for k in n.ALL_KINDS):
            support=n.supports(proposal["predicted"],h)
            attacked=n.blind_attack(source["full_public_input_sha256"],
                                    kind,r,support)
            self.assertEqual(len(attacked),16)


if __name__=="__main__":
    unittest.main()
