"""Independent capacity exhaustions for interface-timed compactification."""
import sys
import unittest
from pathlib import Path
from itertools import combinations
sys.path.insert(0,str(Path(__file__).resolve().parent))
from finite_reference import capacity,signature,witness,finite_audit


class TimeInterfaceCapacityTests(unittest.TestCase):
    def test_registered_resource_vector(self):
        c=capacity(12,6)
        self.assertEqual(c["present_count_classes"],13)
        self.assertEqual(c["delayed_continuation_classes"],4096)
        self.assertEqual(c["delayed_self_contained_bits"],12)
        self.assertEqual(c["extra_bits_with_retained_present_posterior"],10)
        self.assertEqual(c["early_initial_valid_count_states"],7)
        self.assertEqual(c["early_initial_bits"],3)
        self.assertEqual(c["early_peak_count_bits"],4)
        self.assertEqual(c["external_verified_old_sum_message_bits"],3)
        self.assertGreaterEqual(c["external_old_archive_minimum_raw_bits"],12)
        self.assertFalse(c["physical_memory_budget_reported"])

    def test_independently_exhaustive_small_domain(self):
        out=finite_audit(7)
        self.assertGreater(out["same_count_pair_checks"],1000)
        self.assertEqual(out["exact_domains"],21)
        self.assertEqual(out["neural_models_evaluated"],0)

    def test_each_pair_has_same_message_discriminator(self):
        n,j=5,3
        for a,b in combinations(range(1<<n),2):
            if a.bit_count()!=b.bit_count():
                continue
            self.assertEqual(signature(a,n,j)[0],signature(b,n,j)[0])
            self.assertNotEqual(signature(a,n,j),signature(b,n,j))
            s=witness(a,b,n,j)
            self.assertEqual(len(s),j)

    def test_invalid(self):
        for fn in (lambda:capacity(1,1),lambda:capacity(12,12),
                   lambda:capacity(13,3),lambda:capacity(True,1),
                   lambda:signature(-1,4,2),lambda:witness(2,2,4,2)):
            with self.assertRaises(ValueError):fn()


if __name__=="__main__":unittest.main()
