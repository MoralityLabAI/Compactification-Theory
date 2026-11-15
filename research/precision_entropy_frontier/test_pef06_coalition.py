"""PEF-06 independent finite and exact rational coalition/noise oracles."""
import random
from fractions import Fraction
from itertools import product
import unittest

from pef06_coalition import (
    coalition_latch, single_bit_at_zero, disjoint_sensitive_witnesses,
    brute_block_sensitivity_at_zero, globally_averaged_sensitivity,
    local_mobius_coefficients, arbitrary_noise_polynomial,
    analytic_noise_coefficients, evaluate_polynomial,
    exact_zero_noise_probability, exhaustive_zero_noise_probability,
    uniform_noise_disagreement, exhaustive_uniform_noise_disagreement,
    exact_joint_binary_output, independent_joint_binary_output,
    binary_mutual_information, exact_full_noisy_input_mutual_information,
    binary_entropy, audit, suite,
)


class CoalitionNoiseTests(unittest.TestCase):
    def test_zero_individual_influence_and_exact_block_packing(self):
        for m in range(1,6):
            self.assertEqual(single_bit_at_zero(m), 0)
            witnesses=disjoint_sensitive_witnesses(m)
            self.assertEqual(len(witnesses),m)
            self.assertEqual(brute_block_sensitivity_at_zero(m),m)
            for i in range(2*m):
                self.assertEqual(coalition_latch(1<<i,m),0)
            for mask in witnesses:
                self.assertEqual(coalition_latch(mask,m),1)
            for i,j in product(range(2*m),repeat=2):
                if i<j and i//2!=j//2:
                    self.assertEqual(coalition_latch((1<<i)|(1<<j),m),0)

    def test_global_average_sensitivity_is_exactly_m(self):
        for m in range(1,7):
            self.assertEqual(globally_averaged_sensitivity(m),m)
            # Flipping each of the 2m bits changes the function at 11...11.
            all_ones=(1<<(2*m))-1
            self.assertTrue(all(
                coalition_latch(all_ones^ (1<<j),m)!=
                coalition_latch(all_ones,m) for j in range(2*m)
            ))

    def test_general_boolean_mobius_noise_works_for_arbitrary_truth_tables(self):
        rng=random.Random(140119127132)
        for n in range(0,6):
            for _ in range(12):
                values=[rng.randrange(2) for _ in range(1<<n)]
                coeffs=arbitrary_noise_polynomial(values,n)
                for p in (Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2)):
                    direct=sum((
                        Fraction(values[mask])*
                        p**mask.bit_count()*
                        (1-p)**(n-mask.bit_count())
                        for mask in range(1<<n)
                    ),Fraction(0))
                    self.assertEqual(evaluate_polynomial(coeffs,p),direct)

    def test_closed_coalitional_mobius_coefficients_and_independent_false_alarm(self):
        for m in range(1,8):
            direct=local_mobius_coefficients(m)
            self.assertEqual(direct,analytic_noise_coefficients(m))
            self.assertEqual(direct[0],0)
            self.assertEqual(direct[1],0)
            self.assertEqual(direct[2],m)
            for k in range(1,2*m,2):
                self.assertEqual(direct[k],0)
            for p in (Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2)):
                target=exact_zero_noise_probability(m,p)
                self.assertEqual(evaluate_polynomial(direct,p),target)
                self.assertEqual(exhaustive_zero_noise_probability(m,p),target)

    def test_uniform_source_noise_response_independent_oracle(self):
        for m in range(1,5):
            for p in (Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2)):
                self.assertEqual(
                    exhaustive_uniform_noise_disagreement(m,p),
                    uniform_noise_disagreement(m,p)
                )

    def test_exact_2x2_channel_independent_from_enumeration(self):
        for m in range(1,4):
            for p in (Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2)):
                law=exact_joint_binary_output(m,p)
                oracle=independent_joint_binary_output(m,p)
                self.assertEqual(law,oracle)
                self.assertEqual(sum(law.values(),Fraction(0)),1)
                self.assertEqual(law["01"],law["10"])
                self.assertEqual(law["01"]+law["10"],
                                 uniform_noise_disagreement(m,p))

    def test_three_pair_frozen_rational_witness(self):
        p=Fraction(1,4)
        m=3
        self.assertEqual(local_mobius_coefficients(m),(0,0,3,0,-6,0,4))
        self.assertEqual(exact_zero_noise_probability(m,p),Fraction(169,1024))
        self.assertEqual(uniform_noise_disagreement(m,p),Fraction(3367,8192))
        self.assertEqual(exact_joint_binary_output(m,p),{
            "00":Fraction(5849,16384),"01":Fraction(3367,16384),
            "10":Fraction(3367,16384),"11":Fraction(3801,16384)
        })

    def test_information_noisy_full_cube_vs_compressed_output(self):
        for m in range(1,4):
            for p in (Fraction(0),Fraction(1,8),Fraction(1,4),Fraction(1,2)):
                joint=exact_joint_binary_output(m,p)
                Icompressed=binary_mutual_information(joint)
                Ifull=exact_full_noisy_input_mutual_information(m,p)
                Icoord=1-binary_entropy(float(p))
                self.assertLessEqual(Icompressed,Ifull+1e-10)
                self.assertLessEqual(Ifull,Icoord+1e-10)
                self.assertGreaterEqual(Icompressed,-1e-10)
        self.assertAlmostEqual(
            exact_full_noisy_input_mutual_information(3,Fraction(1,4)),
            0.03319598162792792, places=11
        )
        self.assertAlmostEqual(
            binary_mutual_information(
                exact_joint_binary_output(3,Fraction(1,4))
            ),0.0196532318075738,places=11
        )

    def test_suite_and_invalid_parameters(self):
        report=suite()
        self.assertEqual(len(report["exact_rational_cases"]),5)
        self.assertEqual(report["three_distinct_openai_math_papers"],[119,127,132])
        self.assertTrue(all(r["bounded_exhaustive_oracle_checked"]
                            for r in report["exact_rational_cases"]))
        with self.assertRaises(ValueError):
            coalition_latch(-1,3)
        with self.assertRaises(ValueError):
            coalition_latch(1<<6,3)
        with self.assertRaises(ValueError):
            coalition_latch(0,0)
        with self.assertRaises(ValueError):
            audit(3,Fraction(3,4))
        with self.assertRaises(ValueError):
            arbitrary_noise_polynomial([0,1,0],2)


if __name__=="__main__":
    unittest.main()
