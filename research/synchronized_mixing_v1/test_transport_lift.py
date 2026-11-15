"""Regression tests for SMC-02 transport lift. No dependencies beyond stdlib."""
from fractions import Fraction
import unittest
import transport_lift as tl


class LiftTests(unittest.TestCase):
    def test_transposition_group_lumpability_and_uniform_stationarity(self):
        for k in (2, 3):
            base, perms, matchings, R, P = tl.kernel_lift(k)
            self.assertEqual(len(matchings), {2: 3, 3: 15}[k])
            self.assertEqual(len(tl.cross_pairs(base)), 2*k*(k-1))
            for ker, states in ((R, perms), (P, matchings)):
                self.assertTrue(all(sum(row.values()) == tl.ONE for row in ker.values()))
                u = {state: Fraction(1, len(states)) for state in states}
                self.assertEqual(tl.push(u, ker), u)
            for g in matchings:
                self.assertEqual(P[g][g], Fraction(1, 2))
                self.assertEqual(len(P[g]), 1 + k * (k - 1))
                for other, rate in P[g].items():
                    if other != g:
                        self.assertEqual(rate, Fraction(1, 2*k*(k-1)))
            for p, row in R.items():
                g = tl.relabel(p, base)
                aggregate = {}
                for p2, mass in row.items():
                    h = tl.relabel(p2, base)
                    aggregate[h] = aggregate.get(h, Fraction(0)) + mass
                self.assertEqual(aggregate, P[g])

    def test_collision_identity_with_arbitrary_nonneutral_distribution(self):
        outcomes = ('a','b','c','d','e','f')
        mass = {o: Fraction(x, 21) for o, x in zip(outcomes, (1,2,3,4,5,6))}
        f = dict(zip(outcomes, (0,1,0,2,1,2)))
        graph = {}
        for o, p in mass.items():
            graph[f[o]] = graph.get(f[o], Fraction(0)) + p
        joint = {(f[o], o): p for o,p in mass.items()}
        null = {(x,o):px*po for x,px in graph.items() for o,po in mass.items()}
        self.assertEqual(tl.tv(joint, null), tl.ONE - sum(px*px for px in graph.values()))

    def test_graph_spectral_curve_and_optimal_relational_detector(self):
        result = tl.exact_sweep(2, horizon=16)
        self.assertEqual(result['asymptotic_relational_tv'],'2/3')
        for row in result['rows']:
            t = row['t']
            self.assertEqual(Fraction(row['graph_tv']), Fraction(2,3) * Fraction(1,4)**t)
            self.assertEqual(Fraction(row['tv_from_time_matched_product']), Fraction(2,3)*(1-Fraction(1,16)**t))
            self.assertEqual(Fraction(row['graph_collision_probability']), Fraction(1,3)+Fraction(2,3)*Fraction(1,16)**t)
            if t >= 6:
                self.assertEqual(Fraction(row['joint_tv_from_uniform_product']),Fraction(2,3))
        self.assertEqual(Fraction(result['rows'][1]['tv_from_time_matched_product']),Fraction(5,8))

    def test_distinct_permutation_modes_exact_spectral_polynomial(self):
        audit = tl.check_four_slot_spectrum()
        self.assertEqual(audit['roots_with_multiplicity'],{'0':1,'1/4':8,'1/2':6,'3/4':8,'1':1})
        self.assertEqual(audit['trace_moments_0_through_4'],['24','12','15/2','21/4','63/16'])

    def test_equivariant_relabeling_controls(self):
        for k in (2, 3):
            report = tl.gauge_intervention_audit(k, horizon=3)
            self.assertEqual(Fraction(report['stationary_one_side_pass']),Fraction(1,2*k-1))
            for row in report['rows']:
                self.assertEqual(row['simultaneous_relabel_pass'],'1')
        inner=tl.gauge_intervention_audit(2, horizon=4, swap=(0,1))
        self.assertEqual(inner['rows'][0]['graph_only_relabel_pass'],'1')
        cross=tl.gauge_intervention_audit(2, horizon=4, swap=(0,2))
        self.assertEqual(cross['rows'][0]['graph_only_relabel_pass'],'0')

    def test_large_matching_lift(self):
        result=tl.exact_sweep(3,horizon=3)
        self.assertEqual(result['n_graph_states'],15)
        self.assertEqual(result['n_permutation_states'],720)
        self.assertEqual(result['asymptotic_relational_tv'],'14/15')
        for row in result['rows']:
            self.assertEqual(Fraction(row['tv_from_time_matched_product']), tl.ONE-Fraction(row['graph_collision_probability']))


if __name__ == '__main__':
    unittest.main()
