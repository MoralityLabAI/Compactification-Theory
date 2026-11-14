"""MTB-01 finite exact Markov/Brenier/decision oracle regression suite."""
from __future__ import annotations

from fractions import Fraction as F
from math import sqrt
import unittest

from reference import (
    SOURCE_PAPERS, BOX_CELLS, HORIZONS, CUBES, THRESHOLD, MARGIN,
    box_params, refresh_retention, rectangular_w2_squared,
    source_same_transport_squared, one_step_energy,
    cross_transport_squared, independently_integrated_cross_squared,
    box_certificate, probability_above_threshold,
    threshold_flip_probability, threshold_margin_probability,
    margin_flip_certificate, cube_odd_probability, cube_moments,
    hamming_birth_death, hamming_full_cube, hypercube_certificate,
    cube_n_step_bound, source_report,
)


class MarkovBrenierTests(unittest.TestCase):
    def test_refresh_semigroup_and_independent_coordinate_mixture(self):
        for s in (F(0), F(1,100), F(1,4), F(1,2), F(1)):
            for t in range(12):
                self.assertEqual(refresh_retention(s,t),
                                 (1-s)**t)
                self.assertEqual(refresh_retention(s,t+1),
                                 (1-s)*refresh_retention(s,t))
            self.assertEqual(refresh_retention(s,0),1)

    def test_rectangular_brenier_cost_exact(self):
        self.assertEqual(rectangular_w2_squared(1,1,2,1),F(1,3))
        self.assertEqual(rectangular_w2_squared(F(3,2),2,2,F(5,2)),
                         F(1,6))
        self.assertEqual(rectangular_w2_squared(2,3,2,3),0)
        self.assertEqual(rectangular_w2_squared(1,1,F(101,100),1),
                         F(1,30000))

    def test_cross_drift_matches_separate_uniform_integral(self):
        for params in BOX_CELLS:
            for t in HORIZONS:
                a,b,c,d,s=params
                self.assertEqual(
                    cross_transport_squared(*params,t),
                    independently_integrated_cross_squared(*params,t))
                self.assertGreaterEqual(
                    cross_transport_squared(*params,t),
                    rectangular_w2_squared(a,b,c,d))
                self.assertEqual(
                    cross_transport_squared(*params,0),
                    rectangular_w2_squared(a,b,c,d))

    def test_exact_nontrivial_t2_witness(self):
        params=(F(1),F(1),F(2),F(1),F(1,4))
        self.assertEqual(refresh_retention(params[-1],2),F(9,16))
        self.assertEqual(cross_transport_squared(*params,2),F(53,96))
        self.assertEqual(source_same_transport_squared(1,1,F(1,4),2),F(7,48))
        self.assertEqual(one_step_energy(1,1,F(1,4)),F(1,12))
        self.assertLessEqual(F(7,48),2*F(1,12))
        self.assertEqual(threshold_flip_probability(1,2,F(1,4),2),F(23,64))

    def test_hilbert_markov_budget_every_frozen_cell(self):
        for params in BOX_CELLS:
            a,b,c,d,s=params
            for t in HORIZONS:
                row=box_certificate(*params,t)
                self.assertEqual(F(row["one_step_same_map_energy"]),
                                 one_step_energy(a,b,s))
                self.assertLessEqual(
                    F(row["same_map_drift_squared"]),
                    F(row["Hilbert_type2_upper_squared"]))
                self.assertLessEqual(
                    F(row["cross_drift_squared"]),
                    F(row["generic_squared_triangle_upper"]))
                self.assertTrue(row["exact_affine_map_stability_power_one"])

    def test_decision_probabilities_exact_under_no_refresh(self):
        for a,c in ((F(1),F(2)), (F(2),F(1)),
                    (F(1),F(1)),(F(1),F(101,100))):
            expected=abs(probability_above_threshold(a)-
                         probability_above_threshold(c))
            self.assertEqual(threshold_flip_probability(a,c,0,6),expected)
            self.assertEqual(threshold_flip_probability(a,c,F(1,4),0),expected)
        self.assertEqual(threshold_flip_probability(1,F(101,100),0,0),F(1,202))

    def test_maximum_refresh_predicts_independent_threshold_error(self):
        for a,c in ((F(1),F(2)),(F(1),F(101,100)),
                    (F(1,2),F(3))):
            p0=probability_above_threshold(a)
            p1=probability_above_threshold(c)
            expected=p0+p1-2*p0*p1
            self.assertEqual(threshold_flip_probability(a,c,1,1),expected)
        self.assertEqual(threshold_flip_probability(1,1,1,10),F(1,2))

    def test_margin_bound_with_true_label_flip_witness(self):
        for params in BOX_CELLS:
            for t in HORIZONS:
                c=margin_flip_certificate(*params,t)
                self.assertLessEqual(
                    F(c["exact_decision_flip_probability"]),
                    F(c["Markov_margin_flip_upper"]))
                self.assertLessEqual(F(c["Markov_margin_flip_upper"]),1)
        self.assertEqual(threshold_margin_probability(1,F(1,2),F(1,5)),F(2,5))
        cell=margin_flip_certificate(1,1,F(101,100),1,F(1,100),1)
        self.assertLess(F(cell["Markov_margin_flip_upper"]),F(1,2))

    def test_hamming_odd_flip_marginals_and_parity_pairs(self):
        for n in (4,8,16,32):
            for t in (0,1,2,4,8):
                row=cube_moments(n,t)
                self.assertEqual(F(row["mean_hamming"]),
                                 n*cube_odd_probability(n,t))
                self.assertGreaterEqual(
                    F(row["second_hamming_moment"]),0)
                self.assertLessEqual(
                    F(row["Hilbert_squared_displacement"]),t)
                if t==1:
                    self.assertEqual(F(row["second_hamming_moment"]),1)

    def test_hamming_dp_confirms_closed_moments_for_small_cubes(self):
        for n in (4,8,16):
            for t in (0,1,2,3,4,6,8):
                closed=cube_moments(n,t)
                dp=hamming_birth_death(n,t)
                self.assertEqual(dp["mean"],F(closed["mean_hamming"]))
                self.assertEqual(dp["second"],F(closed["second_hamming_moment"]))
                self.assertEqual(sum(dp["pmf"]),1)

    def test_complete_bitmask_random_walk_is_third_independent_oracle(self):
        for n in (4,6,8):
            for t in range(0,7):
                dp=hamming_birth_death(n,t)
                full=hamming_full_cube(n,t)
                self.assertEqual(dp["mean"],full["mean"])
                self.assertEqual(dp["second"],full["second"])
                self.assertLessEqual(full["states"],1<<n)

    def test_hypercube_four_bit_exact_squared_distance(self):
        row=cube_moments(4,4)
        self.assertEqual(row["mean_hamming"],"15/8")
        self.assertEqual(row["second_hamming_moment"],"9/2")
        self.assertEqual(row["L1_Markov_type2_constant_squared_lower"],"9/8")
        self.assertEqual(row["Hilbert_squared_displacement"],"15/8")
        self.assertEqual(row["Hilbert_type2_upper"],4)

    def test_l1_dimension_dependent_markov_lower_bound(self):
        previous=0
        for n in CUBES:
            cert=cube_n_step_bound(n)
            self.assertGreater(cert["markov_type2_constant_lower_float"],previous)
            previous=cert["markov_type2_constant_lower_float"]
            self.assertGreaterEqual(
                cert["markov_type2_constant_lower_float"],
                cert["Jensen_type2_lower_float"])
            self.assertGreaterEqual(cert["Jensen_type3_over2_lower_float"],0)
            self.assertGreaterEqual(
                float(F(cert["E_Hamming"])/n),
                cert["dimension_free_mean_fraction_lower_float"]-1e-14)
        self.assertGreater(previous,4)

    def test_exhaustive_finite_source_report_and_claim_boundaries(self):
        r=source_report()
        self.assertEqual(r["status"],"EXACT_AFFINE_TRANSPORT_MARKOV_CONTROL")
        self.assertEqual(r["source_papers"],[327,374])
        self.assertEqual(len(r["rectangle_cells"]),len(BOX_CELLS)*len(HORIZONS))
        self.assertEqual(len(r["hypercube_n_step_lower_bounds"]),len(CUBES))
        self.assertTrue(r["stationary_reversible_refresh"])
        self.assertFalse(r["source_327_superreflexivity_equivalence_reproved"])
        self.assertFalse(r["source_374_sharp_one_third_theorem_reproved"])
        self.assertFalse(r["trained_neural_checkpoint_evaluated"])

    def test_invalid_inputs_rejected_explicitly(self):
        for params in ((0,1,1,1,F(1,4)),
                       (1,1,-1,1,F(1,4)),
                       (1,1,1,1,F(5,4))):
            with self.assertRaises(ValueError):
                box_params(*params)
        with self.assertRaises(ValueError):
            refresh_retention(F(1,4),-1)
        with self.assertRaises(ValueError):
            cube_moments(3,2)
        with self.assertRaises(ValueError):
            hamming_full_cube(16,4)
        with self.assertRaises(ValueError):
            threshold_margin_probability(1,margin=0)
        with self.assertRaises(ValueError):
            probability_above_threshold(0)
        with self.assertRaises(ValueError):
            box_params(True,1,1,1,F(1,4))


if __name__=="__main__":
    unittest.main()
