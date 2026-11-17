"""Independent exact integer checks for OIC-32, active with Python -O."""
import random
import unittest

from oic32_global import (is_prime, prime_block_word, residue_occupancy,
                          line_weights, pair_moment_identity, minimum_cross_shift_distance,
                          theorem_bounds, reference, choose2)


class OIC32Tests(unittest.TestCase):
    def test_prime_guard(self):
        for n in (0, 1, 4, 6, 8, 9, 15):
            self.assertFalse(is_prime(n))
            with self.assertRaises(ValueError):
                theorem_bounds(n)
        for n in (5, 7, 11, 13):
            self.assertTrue(is_prime(n))
        with self.assertRaises(ValueError):
            theorem_bounds(3)

    def test_global_moment_formula_repeated_points(self):
        for p in (5, 7, 11):
            rng = random.Random(20261111 + p)
            words = [tuple(rng.randrange(p) for _ in range(3*p-1)) for _ in range(10)]
            words.append(tuple([0] * (3*p-1)))
            words.append(tuple((t // p) % p for t in range(3*p-1)))
            for word in words:
                metrics = pair_moment_identity(p, word)
                self.assertEqual(metrics['line_pair_moment'], metrics['identity_rhs'])
                rows = residue_occupancy(p, word)
                self.assertEqual(sum(map(sum, rows)), len(word))
                self.assertEqual(sum(sum(row) for row in line_weights(p, word)), p*len(word))

    def test_residue_pairs_for_forbidden_horizon(self):
        for p in (5, 7, 11, 13):
            word = tuple(t % p for t in range(3*p - 1))
            metrics = pair_moment_identity(p, word)
            expected = (9*p*p - 15*p + 6)//2
            self.assertEqual(metrics['different_residue_pairs'], expected)

    def test_strict_analytic_gap_and_21_query_case(self):
        for p in (5, 7, 11, 13, 17, 19, 23):
            b = theorem_bounds(p)
            self.assertEqual(b['minimum_query_horizon'], 3*p)
            self.assertGreater(b['integer_contradiction_gap'], 0)
            self.assertEqual(b['line_pairs_lower_all_words'] -
                             b['line_pairs_upper_if_robust'], b['integer_contradiction_gap'])
        b = theorem_bounds(7)
        self.assertEqual(b['line_pairs_lower_all_words'], 171)
        self.assertEqual(b['line_pairs_upper_if_robust'], 139)
        self.assertEqual(b['integer_contradiction_gap'], 32)

    def test_attaining_code_all_marker_pairs(self):
        for p in (5, 7, 11, 13):
            word = prime_block_word(p)
            self.assertEqual(len(word), 3*p)
            self.assertEqual(set(word), {0, 1, 2})
            self.assertEqual(minimum_cross_shift_distance(p, word), 3)
            metrics = pair_moment_identity(p, word)
            self.assertEqual(metrics['line_pair_moment'], metrics['identity_rhs'])

    def test_random_words_below_bound_have_collision(self):
        rng = random.Random(3212026)
        for p, trials in ((5, 70), (7, 50), (11, 15)):
            for _ in range(trials):
                word = tuple(rng.randrange(p) for _ in range(3*p - 1))
                self.assertLessEqual(minimum_cross_shift_distance(p, word), 2)

    def test_machine_receipt(self):
        data = reference()
        self.assertEqual([row['prime'] for row in data['rows']], [5, 7, 11, 13])
        self.assertTrue(data['excludes_full_marker_exception'])
        self.assertTrue(all(x['attaining_minimum_cross_shift_distance']==3
                            for x in data['rows']))


if __name__ == '__main__':
    unittest.main()
