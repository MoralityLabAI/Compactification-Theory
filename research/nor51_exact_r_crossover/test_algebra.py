"""NOR51 source-free controls: only old/toy inputs, never the 48 new sources."""
from __future__ import annotations
import random,unittest
from research.nor45_authorized_program import independent_execution_verifier as native
from research.nor46_exact_integer_trace import trace_verifier as nor46
from research.nor47_gmp_exact_convolution import gmp_kernel as gmp
from research.nor51_exact_r_crossover import algebra,cohort

class NewtonExactAlgebraTests(unittest.TestCase):
    def test_independent_quadratic_agreement_and_merkle_root(self):
        saved=nor46.exact_convolution_prefix
        for seed in (3,17,211):
            rng=random.Random(seed)
            for length in (1,2,3,5,17,31,65,127):
                src=(1,)+tuple(rng.choice((-3,-2,-1,0,0,1,2,3))
                                for _ in range(length-1))
                direct=algebra.native_single_inverse(src)
                newton=algebra.newton_inverse(src)
                self.assertEqual(newton,direct)
                self.assertEqual(algebra.checked_inverse_witness(src,newton),newton)
                a=algebra.direct_with_root(src)
                b=algebra.gmp_witness_with_root(src,direct)
                c=algebra.newton_verified_with_root(src)
                self.assertEqual(a,b)
                self.assertEqual(b,c)
                self.assertEqual(a[1],native.independently_computed_root(direct))
        self.assertIs(saved,nor46.exact_convolution_prefix)

    def test_wrong_witness_and_invalid_input_rejected(self):
        src=(1,2,-1,0,3,-4,5,0,1,1,-3,2)
        inv=algebra.native_single_inverse(src)
        self.assertEqual(algebra.checked_inverse_witness(src,inv),inv)
        wrong=list(inv);wrong[-1]+=1
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness(src,wrong)
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness(src,inv[:-1])
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness(src,["1",*inv[1:]])
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness(src,[True,*inv[1:]])
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness((2,*src[1:]),inv)
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness((1,True,*src[2:]),inv)
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness((1,1<<(nor46.MAX_COEFFICIENT_BITS+1)),(1,0))
        with self.assertRaises((ValueError,TypeError)):
            algebra.checked_inverse_witness((1,0),(1,1<<(nor46.MAX_COEFFICIENT_BITS+1)))

    def test_exact_signed_gmp_and_python_convolution_small_controls(self):
        for a,b in [
            ((1,-2,3,-4),(1,3,-1,-3)),
            ((1,0,0,0,1),(1,0,0,0,-1)),
            ((1,),(1,)),
            ((1,1<<256,-7),(1,-(1<<255),6)),
        ]:
            expected=nor46.exact_convolution_prefix(a,b)
            self.assertEqual(gmp.full_gmp(a,b),expected)
        with self.assertRaises((ValueError,TypeError)):
            gmp.full_gmp((1,0),(1,))
        with self.assertRaises((ValueError,TypeError)):
            gmp.full_gmp((1,True),(1,0))

    def test_prior_original_archive_and_analytic_first_grid_resource_bounds(self):
        from research.nor50_memoized_exact import original_archive_audit as archived
        assert archived.audit()["first_verdict"]=="FAIL 4/5"
        self.assertEqual(cohort.REPS,(0,1))
        self.assertEqual(cohort.LENGTHS,(769,1537,3073))
        self.assertEqual(cohort.PROFILES,("s4-b2","s4-b8","s4-b16","s1-b2"))
        self.assertEqual(cohort.FAMILIES,("signed_random","alternating"))
        self.assertEqual(cohort.REGISTERED,"83fabe1c34c2bedbba35faa928349f41b964115a")
        for m in cohort.LENGTHS:
            for profile,(stride,bits) in cohort.PROFILE_PARAMETERS.items():
                reduced=(m-1)//stride
                upper_recip_bits=reduced*bits+1
                worst_pack=m*(2*upper_recip_bits+(4*m).bit_length()+2)
                self.assertLessEqual(upper_recip_bits,nor46.MAX_COEFFICIENT_BITS)
                self.assertLess(worst_pack,nor46.MAX_PACKED_BITS)

if __name__=="__main__":
    unittest.main()
