"""MTB-01B: stationary nonreversible Hilbert drift controls."""
from __future__ import annotations

from fractions import Fraction as F
import unittest
from nonreversible import (
    SIZES, square_point, sqdist, perimeter_drift, quarter_drift,
    detailed_balance_violation, report,
)


class DirectedSquareDriftTests(unittest.TestCase):
    def test_all_perimeter_edges_have_unit_squared_length(self):
        for m in SIZES:
            pts=[square_point(m,i) for i in range(4*m)]
            self.assertEqual(len(set(pts)),4*m)
            self.assertEqual(pts[0],(0,0))
            self.assertEqual(pts[m],(m,0))
            self.assertEqual(pts[2*m],(m,m))
            self.assertEqual(pts[3*m],(0,m))
            for i in range(4*m):
                self.assertEqual(sqdist(pts[i],pts[(i+1)%(4*m)]),1)

    def test_stationary_uniform_law_but_detailed_balance_fails(self):
        for m in SIZES:
            c=detailed_balance_violation(m)
            self.assertTrue(c["all_states_equal_indegree_outdegree"])
            self.assertTrue(c["detailed_balance_rejected"])
            self.assertEqual(c["forward_stationary_flux"],str(F(1,4*m)))
            self.assertEqual(c["reverse_stationary_flux"],"0")

    def test_independent_exact_perimeter_sum_recovers_quarter_formula(self):
        for m in SIZES:
            self.assertEqual(perimeter_drift(m,m),F(2*m*m+1,3))
            r=quarter_drift(m)
            self.assertEqual(r["quarter_drift_squared"],str(F(2*m*m+1,3)))
            self.assertEqual(r["one_step_squared_energy"],"1")
            self.assertGreater(F(r["observed_squared_ratio_to_reversible_budget"]),1)

    def test_stationary_drift_breaks_reversible_Hilbert_budget(self):
        for m in SIZES:
            base=perimeter_drift(m,1)
            self.assertEqual(base,1)
            self.assertGreater(perimeter_drift(m,m),m*base)
            self.assertEqual(F(quarter_drift(m)["observed_squared_ratio_to_reversible_budget"]),
                             F(2*m*m+1,3*m))

    def test_rational_square_quarter_positions(self):
        m=4
        self.assertEqual(square_point(m,1),(1,0))
        self.assertEqual(square_point(m,1+m),(4,1))
        self.assertEqual(sqdist((1,0),(4,1)),10)
        self.assertEqual(perimeter_drift(m,m),F(33,3))

    def test_time_periodicity_and_zero_drift(self):
        for m in (2,4,8,16):
            self.assertEqual(perimeter_drift(m,0),0)
            self.assertEqual(perimeter_drift(m,4*m),0)
            for t in range(0,min(4*m,10)):
                self.assertEqual(perimeter_drift(m,t),
                                 perimeter_drift(m,t+4*m))

    def test_control_source_scope_and_fingerprints(self):
        r=report()
        self.assertEqual(r["source_papers"],[327,374])
        self.assertTrue(r["independent_postprotocol_extension"])
        self.assertFalse(r["source_327_markov_type_applied_to_this_chain"])
        self.assertFalse(r["source_374_brenier_map_needed_for_this_control"])
        self.assertEqual(len(r["rows"]),len(SIZES))
        self.assertEqual(len(r["source_sha256"]),64)

    def test_invalid_indices_and_side_lengths_fail(self):
        for m,i in ((1,0),(2,-1),(2,8),(3,12),(0,0)):
            with self.assertRaises(ValueError):
                square_point(m,i)
        with self.assertRaises(ValueError):
            perimeter_drift(2,-1)
        with self.assertRaises(ValueError):
            detailed_balance_violation(1)


if __name__=="__main__":
    unittest.main()
