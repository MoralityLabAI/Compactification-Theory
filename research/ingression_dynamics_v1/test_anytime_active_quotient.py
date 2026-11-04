"""QO-07 independent finite oracles for anytime active quotient certificates."""
from collections import defaultdict
from fractions import Fraction as Q
from itertools import product
from math import comb
import unittest

from research.ingression_dynamics_v1.anytime_active_quotient import (
    ProbeLedger, anytime_outer_interval, mixture_evalue, validate_bounds,
    partition_interval_bounds, robust_optimum, certificate, choose_probe,
    frozen_error, run_synthetic, hidden_alias_word_probability,
)
from research.ingression_dynamics_v1.rate_sampling import partition_radius
from research.ingression_dynamics_v1.goal_aware_partition import partitions


class AnytimeMixtureTests(unittest.TestCase):
    def test_fixed_time_mixture_likelihood_is_exact_martingale_mean(self):
        for p in (Q(1, 4), Q(1, 2), Q(3, 4)):
            for t in range(1, 9):
                expectation = sum((
                    comb(t, s) * p**s * (1-p)**(t-s) *
                    mixture_evalue(t, s, p) for s in range(t+1)
                ), Q(0))
                self.assertEqual(expectation, Q(1))
                for s in range(t):
                    now = mixture_evalue(t, s, p)
                    next_expected = (p * mixture_evalue(t+1, s+1, p) +
                                     (1-p) * mixture_evalue(t+1, s, p))
                    self.assertEqual(now, next_expected)

    def test_interval_outer_brackets_contain_every_accepted_grid_point(self):
        alpha = Q(1, 10)
        for t in range(1, 15):
            for s in range(t+1):
                lo, hi = anytime_outer_interval(t, s, alpha, 3, 10)
                self.assertLessEqual(lo, Q(s, t))
                self.assertGreaterEqual(hi, Q(s, t))
                for j in range(41):
                    p = Q(j, 40)
                    if p**s * (1-p)**(t-s) >= alpha / (
                            3*(t+1)*comb(t, s)):
                        self.assertLessEqual(lo, p)
                        self.assertGreaterEqual(hi, p)

    def test_exact_crossing_probability_at_every_time(self):
        """Independent binomial-count recursion, not stochastic simulation."""
        for p in (Q(1, 2), Q(1, 4)):
            survivors = {0: Q(1)}
            failure = Q(0)
            alpha = Q(1, 10)
            for t in range(1, 21):
                next_mass = defaultdict(Q)
                for successes, probability in survivors.items():
                    next_mass[successes] += probability * (1-p)
                    next_mass[successes+1] += probability * p
                survivors = {}
                for successes, probability in next_mass.items():
                    lo, hi = anytime_outer_interval(t, successes, alpha, 1, 10)
                    if lo <= p <= hi:
                        survivors[successes] = probability
                    else:
                        failure += probability
                self.assertLessEqual(failure, alpha)
                self.assertEqual(failure+sum(survivors.values(), Q(0)), Q(1))
            self.assertGreater(failure, Q(0))

    def test_ledger_monotone_and_invalid_updates(self):
        ledger = ProbeLedger(2, 2, Q(1, 20), 10)
        initial = ledger.snapshot()
        self.assertEqual(initial[0][1], (Q(0), Q(1)))
        for o in (0, 1, 0, 0, 0, 1, 1, 0):
            before = ledger.snapshot()[1][0]
            after = ledger.observe(1, 0, o)
            self.assertGreaterEqual(after[0], before[0])
            self.assertLessEqual(after[1], before[1])
        self.assertEqual(ledger.query_count(), 8)
        for args in ((2,0,1), (0,2,0), (0,0,2), (True,0,0)):
            with self.assertRaises(ValueError):
                ledger.observe(*args)
        self.assertEqual(ledger.query_count(), 8)

    def test_zero_and_inconsistent_nulls(self):
        self.assertEqual(mixture_evalue(0, 0, Q(1, 2)), Q(1))
        self.assertIsNone(mixture_evalue(1, 1, Q(0)))
        self.assertEqual(anytime_outer_interval(0, 0), (Q(0), Q(1)))
        with self.assertRaises(ValueError):
            anytime_outer_interval(2, 3)


class RobustQuotientTests(unittest.TestCase):
    def test_exact_minimax_over_interval_endpoints_and_midpoint_grid(self):
        """Independent oracle scans rational macro centers on a 1/16 grid."""
        candidates = ((Q(0), Q(0)), (Q(0), Q(1, 2)),
                      (Q(1, 4), Q(3, 4)), (Q(1, 2), Q(1)),
                      (Q(0), Q(1)))
        count = 0
        for x,y,z in product(candidates, repeat=3):
            bounds = ((x,), (y,), (z,))
            for labels in ((0,0,0), (0,1,0), (0,1,2)):
                radius, lower, centers = partition_interval_bounds(bounds, labels)
                direct = max(
                    min(
                        max(max(abs(lo - Q(j,16)), abs(hi - Q(j,16)))
                            for i, row in enumerate(bounds)
                            if labels[i] == group
                            for lo,hi in row)
                        for j in range(17))
                    for group in set(labels))
                self.assertEqual(radius, direct)
                self.assertGreaterEqual(radius, lower)
                count += 1
        self.assertEqual(count, 375)

    def test_robust_global_optimum_against_direct_partition_enumeration(self):
        cases = (
            (((Q(0),Q(1)),), ((Q(0),Q(1)),)),
            (((Q(0),Q(1,4)),), ((Q(3,4),Q(1)),)),
            (((Q(0),Q(1,8)), (Q(1,2),Q(3,4))),
             ((Q(1,4),Q(1,2)), (Q(3,4),Q(1)))),
        )
        for bounds in cases:
            for k in range(1,len(bounds)+1):
                U,L,labels,centers=robust_optimum(bounds,k)
                directly=[]
                for p in partitions(tuple(range(len(bounds)))):
                    if len(set(p.values()))<=k:
                        directly.append(partition_interval_bounds(bounds,
                            tuple(p[i] for i in range(len(bounds)))))
                self.assertEqual(U,min(c[0] for c in directly))
                self.assertEqual(L,min(c[1] for c in directly))
                self.assertEqual(U,partition_interval_bounds(bounds,labels)[0])
                self.assertLessEqual(U-L,max(hi-lo for row in bounds for lo,hi in row))

    def test_absent_probe_blocks_subhalf_uniform_certificate(self):
        ledger = ProbeLedger(3, 2)
        for _ in range(50):
            ledger.observe(0,0,0)
        report = certificate(ledger.snapshot(),2,Q(1,4))
        self.assertEqual(report["decision"],"undetermined")
        self.assertEqual(report["robust_macro_error_upper"],Q(1,2))
        self.assertEqual(report["true_optimum_lower"],Q(0))
        self.assertEqual(report["maximum_interval_width"],Q(1))

    def test_robust_certificate_allows_optional_stopping(self):
        ledger = ProbeLedger(2, 1, Q(1, 20), 10)
        for _ in range(50):
            ledger.observe(0,0,0)
            ledger.observe(1,0,1)
        before = certificate(ledger.snapshot(),2,Q(1,4))
        self.assertEqual(before["decision"],"certified-feasible")
        self.assertLessEqual(before["robust_macro_error_upper"],Q(1,4))
        self.assertEqual(before["mutable_label_bits"],2)
        # Frozen macro predictions are audited using true digital rates.
        self.assertLessEqual(frozen_error(((Q(0),),(Q(1),)),
            before["transient_labels"],before["frozen_midpoints"]),
            before["robust_macro_error_upper"])

    def test_robust_rejection_from_separated_intervals(self):
        bounds = (((Q(0),Q(1,10)),),
                  ((Q(9,10),Q(1)),))
        report = certificate(bounds,1,Q(1,4))
        self.assertEqual(report["decision"],"certified-infeasible")
        self.assertEqual(report["true_optimum_lower"],Q(2,5))
        self.assertEqual(report["robust_macro_error_upper"],Q(1,2))

    def test_seeded_active_and_round_robin_receipts(self):
        rates = ((Q(0),Q(1,4)),(Q(1,2),Q(1,2)),(Q(1),Q(3,4)))
        results = []
        for policy in ("round_robin","width","bottleneck"):
            x=run_synthetic(rates,90,2,policy=policy,seed=19)
            y=run_synthetic(rates,90,2,policy=policy,seed=19)
            self.assertEqual(x,y)
            self.assertEqual(x["observed_queries"],90)
            self.assertEqual(sum(map(sum,x["counts"])),90)
            self.assertTrue(x["covered_true_rates"])
            self.assertLessEqual(x["true_frozen_error"],
                                 x["robust_macro_error_upper"])
            results.append(x)
        self.assertTrue(all(r["mutable_label_bits"]<=2 for r in results))

    def test_hidden_redraw_indistinguishability(self):
        worlds = ((Q(0),Q(1)), (Q(1,2),Q(1,2)))
        for t in range(9):
            for word in product((0,1),repeat=t):
                values = [hidden_alias_word_probability(w,word) for w in worlds]
                self.assertEqual(values[0],values[1])
                self.assertEqual(values[0],Q(1,2**t))
        # Their exact goal-visible quotient sizes are different.
        self.assertEqual(partition_radius(((Q(0),),(Q(1),)),(0,0)),Q(1,2))
        self.assertEqual(partition_radius(((Q(1,2),),(Q(1,2),)),(0,0)),Q(0))

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):
            ProbeLedger(0,2)
        with self.assertRaises(ValueError):
            validate_bounds((((Q(3,4),Q(1,4)),),))
        with self.assertRaises(ValueError):
            robust_optimum((((Q(0),Q(1)),),),0)
        with self.assertRaises(ValueError):
            choose_probe(ProbeLedger(1,1),1,policy="oracle")

if __name__ == "__main__":
    unittest.main()
