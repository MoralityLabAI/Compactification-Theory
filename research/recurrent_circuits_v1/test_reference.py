import itertools
import random
import unittest
from fractions import Fraction as Q

from reference import (Affine, apply_rows, coordinates, error_envelope, finite_partition,
    invariant_tube, label_coverage_bound, minimal_affine_quotient, restore,
    restored_nand, roundtrip_bytes, row_basis)


class QuotientTests(unittest.TestCase):
    def test_direct_read_and_identity_need_one_bit(self):
        f = Affine(tuple(1 << i for i in range(8)))
        self.assertEqual(minimal_affine_quotient(8, [f], [1]).bits, 1)

    def test_cyclic_reads_need_every_bit(self):
        f = Affine(tuple(1 << ((i + 1) % 8) for i in range(8)))
        self.assertEqual(minimal_affine_quotient(8, [f], [1]).bits, 8)

    def test_parity_with_permutation_needs_one_bit(self):
        f = Affine(tuple(1 << ((i + 1) % 8) for i in range(8)))
        self.assertEqual(minimal_affine_quotient(8, [f], [255]).bits, 1)

    def test_parity_with_overwrite_interventions_needs_all_bits(self):
        for m in range(1, 9):
            maps = [Affine(tuple(0 if i == j else 1 << i for i in range(m)), bias)
                    for j in range(m) for bias in (0, 1 << j)]
            self.assertEqual(minimal_affine_quotient(m, maps, [(1 << m)-1]).bits, m)

    def test_zero_output_and_zero_dimension(self):
        self.assertEqual(minimal_affine_quotient(3, [Affine((1, 2, 4))], []).bits, 0)
        q = minimal_affine_quotient(0, [Affine(())], [])
        self.assertEqual((q.encode(0), q.readout(0), q.transitions[0](0)), (0, 0, 0))

    def test_translation_is_retained(self):
        q = minimal_affine_quotient(2, [Affine((1, 2), 1)], [1])
        self.assertEqual(q.transitions[0](0), 1)
        self.assertEqual(q.transitions[0](1), 0)

    def test_rref_is_canonical(self):
        self.assertEqual(row_basis((3, 5, 6), 3), row_basis((6, 5, 3, 3, 0), 3))
        self.assertEqual(row_basis((1, 2, 4), 3), (1, 2, 4))

    def test_coordinate_rejection(self):
        with self.assertRaises(ValueError): coordinates(2, (1,), 2)

    def test_random_systems_against_independent_partition(self):
        rng = random.Random(20261029)
        for trial in range(160):
            m = 1 + trial % 5
            maps = tuple(Affine(tuple(rng.randrange(1 << m) for _ in range(m)),
                                rng.randrange(1 << m)) for _ in range(trial % 4))
            outputs = tuple(rng.randrange(1 << m) for _ in range(trial % 3))
            q = minimal_affine_quotient(m, maps, outputs)
            blocks = finite_partition(m, maps, outputs)
            self.assertEqual(1 << q.bits, len(set(blocks)))
            for x in range(1 << m):
                self.assertEqual(q.readout(q.encode(x)), apply_rows(outputs, x))
                for f, g in zip(maps, q.transitions):
                    self.assertEqual(q.encode(f(x)), g(q.encode(x)))
                for y in range(1 << m):
                    self.assertEqual(q.encode(x) == q.encode(y), blocks[x] == blocks[y])

    def test_validation(self):
        for rows, bias in [((True,), 0), ((2,), 0), ((1,), 2), ((-1,), 0)]:
            with self.assertRaises(ValueError): Affine(rows, bias)
        with self.assertRaises(ValueError): minimal_affine_quotient(2, [Affine((1,))], [1])
        with self.assertRaises(ValueError): finite_partition(13, [], [])


class RobustnessTests(unittest.TestCase):
    def test_all_nand_margin_corners_and_interior(self):
        for x, y in itertools.product((0, 1), repeat=2):
            for ex, ey in itertools.product((Q(-1,8), 0, Q(1,8)), repeat=2):
                self.assertEqual(restored_nand(x+ex, y+ey), 1-x*y)

    def test_margin_outside_certificate_can_fail(self):
        self.assertNotEqual(restored_nand(1+Q(1,4), Q(1,4)), 1)

    def test_restore_all_memory_and_recurrent_circuit(self):
        # Parallel NAND ring, including uniform post-step bounded perturbations.
        for initial in itertools.product((0, 1), repeat=4):
            actual = tuple(map(Q, initial))
            symbolic = initial
            for t in range(128):
                next_symbolic = tuple(1-symbolic[i]*symbolic[(i-1)%4] for i in range(4))
                actual = tuple(restored_nand(actual[i], actual[(i-1)%4])
                               + Q(1 if (t+i)%2 else -1,8) for i in range(4))
                self.assertEqual(tuple(int(x >= Q(1,2)) for x in actual), next_symbolic)
                symbolic = next_symbolic

    def test_invariant_tube_boundary(self):
        self.assertTrue(invariant_tube(Q(1,2), Q(1,16), Q(1,8)))
        self.assertFalse(invariant_tube(Q(1,2), Q(1,16)+Q(1,1000), Q(1,8)))
        self.assertFalse(invariant_tube(1, 0, Q(1,8)))
        self.assertFalse(invariant_tube(0, 0, Q(1,2)))

    def test_error_envelope_including_unstable(self):
        self.assertEqual(error_envelope(0, Q(1,64), 1, 16), Q(1,4))
        self.assertEqual(error_envelope(0, 1, 2, 5), 31)
        self.assertEqual(error_envelope(Q(1,8), Q(1,16), Q(1,2), 100), Q(1,8))
        self.assertEqual(error_envelope(1, 2, 3, 0), 1)

    def test_noise_and_restoration_validation(self):
        for x in (True, 0.1, float('nan')):
            with self.assertRaises(TypeError): restore(x)
        for a in (0, Q(1,2), 1):
            with self.assertRaises(ValueError): restore(0, a)
        with self.assertRaises(ValueError): error_envelope(0, -1, 1, 2)


class ResourceTests(unittest.TestCase):
    def test_random_label_counting_and_tolerated_errors(self):
        self.assertEqual(label_coverage_bound(16, 4), Q(1,4096))
        self.assertEqual(label_coverage_bound(16, 4, choices=4), Q(1,1024))
        self.assertEqual(label_coverage_bound(16, 4, allowed_errors=1), Q(17,4096))
        self.assertEqual(label_coverage_bound(4, 8), 1)
        self.assertEqual(label_coverage_bound(0, 0), 1)

    def test_roundtrip_accounting(self):
        self.assertEqual(roundtrip_bytes(1024, 16, 1), 32768)
        self.assertEqual(roundtrip_bytes(1024, 16, 4), 8192)
        self.assertEqual(roundtrip_bytes(1024, 17, 4), 10240)

    def test_resource_validation(self):
        with self.assertRaises(ValueError): label_coverage_bound(4, 1, allowed_errors=5)
        with self.assertRaises(ValueError): label_coverage_bound(4, True)
        with self.assertRaises(ValueError): roundtrip_bytes(1, 0, 1)
        with self.assertRaises(ValueError): roundtrip_bytes(1, 1, True)


if __name__ == '__main__':
    unittest.main()
