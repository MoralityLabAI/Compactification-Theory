"""FPB-01 exact finite Fourier and coordinated-block intervention checks."""
from __future__ import annotations

import itertools
import unittest
from fractions import Fraction as Q
from reference import (
    BASE_N, HIDDEN_N, SENTINEL, base_fourier_numerator, base_sensitivity,
    character, exact_base_rejection_count, exact_controls, hidden17,
    hidden_fourier, hidden_sensitivity, hidden_sign, lifted_fourier,
    lifted_sign, majority5_fourier, majority5_sign, rubinstein16,
    selector_global_singleton_sum, selector_sign, sensitive_zero_blocks,
    smallest_parity_lift_check, adaptive_unpaired_transcript,
    conditional_unpaired_law, conditional_completed_pair_law,
    adaptive_probe_control,
)


class FiniteFourierProbes(unittest.TestCase):
    def test_rubinstein_16_source_counts_and_symmetry(self):
        self.assertEqual(exact_base_rejection_count(),38416)
        self.assertEqual(rubinstein16(0),0)
        self.assertEqual(rubinstein16(0b0011),1)
        self.assertEqual(rubinstein16(0b1100),1)
        self.assertEqual(rubinstein16(0b0101),0)
        for x in (0, 1, 3, 12, 5432, 65535):
            self.assertEqual(rubinstein16(x),
                             rubinstein16(x ^ ((1<<BASE_N)-1)))

    def test_exact_all_state_sensitivity(self):
        r=base_sensitivity()
        self.assertEqual(r["states_exhausted"],1<<BASE_N)
        self.assertEqual(r["sensitivity_zero_max"],4)
        self.assertEqual(r["sensitivity_one_max"],4)
        b=hidden_sensitivity()
        self.assertEqual(b["global_hidden17"],5)
        self.assertEqual(b["global_parity_lift"],10)

    def test_disjoint_coordinated_zero_interventions(self):
        masks=sensitive_zero_blocks()
        self.assertEqual(len(masks),9)
        self.assertEqual(sum(x.bit_count() for x in masks),17)
        self.assertEqual(hidden17(0),0)
        self.assertTrue(all(hidden17(mask)==1 for mask in masks))
        seen=0
        for mask in masks:
            self.assertFalse(seen & mask)
            seen |= mask
            self.assertEqual(lifted_sign(mask,0),-lifted_sign(0,0))
        self.assertEqual(seen,(1<<HIDDEN_N)-1)

    def test_singleton_lift_coefficients_are_exactly_zero(self):
        for i in range(HIDDEN_N):
            self.assertEqual(lifted_fourier(1<<i,0),0)
            self.assertEqual(lifted_fourier(0,1<<i),0)
        self.assertEqual(lifted_fourier(SENTINEL,SENTINEL),
                         Q(2401,4096))

    def test_hidden_degree_and_fourth_order_pair_feature(self):
        full_mask=(1<<HIDDEN_N)-1
        self.assertEqual(hidden_fourier(full_mask),Q(1,4096))
        self.assertEqual(lifted_fourier(full_mask,full_mask),Q(1,4096))
        self.assertEqual(hidden_fourier(3),Q(-343,4096))
        self.assertEqual(lifted_fourier(3,3),Q(-343,4096))
        self.assertEqual(lifted_fourier(3,1),0)

    def test_mean_bias_and_sentinel_exact_rationals(self):
        self.assertEqual(hidden_fourier(0),Q(-1695,4096))
        self.assertEqual(hidden_fourier(SENTINEL),Q(2401,4096))
        for i in range(BASE_N):
            self.assertEqual(hidden_fourier(1<<i),0)
        self.assertEqual(base_fourier_numerator((1<<BASE_N)-1),32)

    def test_independent_exhaustive_tiny_transport(self):
        x=smallest_parity_lift_check()
        self.assertEqual(x["base_boolean_functions"],16)
        self.assertTrue(x["equal_mask_transport_verified"])
        self.assertTrue(x["singleton_coefficients_zero"])

    def test_paired_toggle_changes_z_syndrome_not_output(self):
        pairs=[(0,0),(1,3),(12345,54321),(131071,0)]
        for x,y in pairs:
            for i in (0,2,9,16):
                self.assertEqual(lifted_sign(x,y),
                                 lifted_sign(x ^ (1<<i),y ^ (1<<i)))
        # The all-zero lift is sensitive to the two individual
        # sentinel flips, though a matched flip cancels.
        self.assertNotEqual(lifted_sign(0,0),lifted_sign(SENTINEL,0))
        self.assertNotEqual(lifted_sign(0,0),lifted_sign(0,SENTINEL))
        self.assertEqual(lifted_sign(0,0),lifted_sign(SENTINEL,SENTINEL))

    def test_explicit_surface_selector_spectrum(self):
        surface=sum((majority5_fourier(1<<i) for i in range(5)),Q())
        self.assertEqual(surface,Q(15,8))
        self.assertEqual(majority5_fourier(0),0)
        self.assertEqual(selector_global_singleton_sum(),Q(9375,8192))
        self.assertEqual(selector_sign(0,0,0,0),majority5_sign(0))
        self.assertEqual(selector_sign(1,0,0,0),lifted_sign(0,0))
        self.assertEqual(selector_sign(1,31,SENTINEL,0),
                         lifted_sign(SENTINEL,0))

    def test_regression_population_exact_mse(self):
        mu=hidden_fourier(0)
        c=hidden_fourier(SENTINEL)
        self.assertEqual(1-mu**2-(1-mu**2-c**2),c**2)
        self.assertGreater(c**2,Q(1,3))
        self.assertGreater(1-mu**2-c**2,0)

    def test_unpaired_adaptive_transcript_exact_no_information(self):
        for n in (1,2,3,4,5):
            reference=conditional_unpaired_law(n,0)
            self.assertEqual(len(reference),1<<n)
            self.assertEqual(sum(reference.values()),1)
            for z in range(1<<n):
                self.assertEqual(conditional_unpaired_law(n,z),reference)
        report=adaptive_probe_control(5)
        self.assertEqual(report["conditional_source_states_checked"],62)
        self.assertTrue(report["adaptive_unpaired_transcripts_source_independent"])

    def test_single_completed_pair_proves_one_hidden_bit(self):
        for n in range(1,6):
            for i in range(n):
                for z in range(1<<n):
                    law=conditional_completed_pair_law(n,z,i)
                    self.assertEqual(sum(law.values()),1)
                    for (a,b),prob in law.items():
                        self.assertEqual(a^b,(z>>i)&1)
                        self.assertEqual(prob,Q(1,2))
            self.assertFalse(
                set(conditional_completed_pair_law(n,0)) &
                set(conditional_completed_pair_law(n,1)))

    def test_adaptive_probe_side_selection_uses_prior_result(self):
        # First x_0 determines whether index 1 is read from x or y.
        self.assertEqual(adaptive_unpaired_transcript(2,0,3),
                         ((0,0,0),(0,1,0)))
        self.assertEqual(adaptive_unpaired_transcript(2,1,2),
                         ((0,0,1),(1,1,1)))

    def test_frozen_source_reference_receipt(self):
        r=exact_controls()
        self.assertEqual(r["status"],"EXACT_FINITE_FOURIER_CONTROL")
        self.assertEqual(r["selector_fourier_degree"],35)
        self.assertEqual(r["hidden_probe_pair_mse_gain"],
                         str(Q(2401,4096)**2))
        self.assertFalse(r["trained_transformer_checkpoint_tested"])
        self.assertFalse(r["openai_132_asymptotic_witness_reproduced"])
        self.assertFalse(r["openai_192_unbounded_ratio_witness_reproduced"])

    def test_validation_rejects_invalid_registers(self):
        for bad in (-1, 1<<BASE_N, 1.1, True):
            with self.assertRaises(ValueError):
                rubinstein16(bad)
        for bad in (-1, 1<<HIDDEN_N):
            with self.assertRaises(ValueError):
                hidden17(bad)
        with self.assertRaises(ValueError):
            lifted_sign(1<<HIDDEN_N,0)
        with self.assertRaises(ValueError):
            hidden_fourier(1<<HIDDEN_N)


if __name__ == "__main__":
    unittest.main()
