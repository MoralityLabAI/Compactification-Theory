"""SCM-04 exact triple-linking tests; all fixtures are mathematical/synthetic."""
from __future__ import annotations

import pathlib
import random
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from knot import extract_closed_braid, normalized_alexander, synthetic_braid
from nonabelian import ArtinWordBudgetExceeded, artin_action
from milnor import (
    PureThreeBraidRequired,
    mixed_magnus_coefficient,
    normalized_conjugator,
    pure_linking_numbers,
    triple_milnor,
    token_order_milnor,
    triple_milnor_nilpotent,
    cubic_artin_images,
)
from rmp_capture import analyze_cycle, compare_double_resolution

A = (1, 1)
B = (2, 2)
C = A + B + (-1, -1) + (-2, -2)
CI = tuple(-v for v in reversed(C))


def direct_magnus_degree_two(word):
    """Independent slow noncommutative polynomial product through degree two."""
    polynomial = {(): 1}
    for letter in word:
        k = abs(letter)
        elementary = {(): 1, (k,): 1 if letter > 0 else -1}
        if letter < 0:
            elementary[(k, k)] = 1
        product = defaultdict(int)
        for prefix, a in polynomial.items():
            for suffix, b in elementary.items():
                if len(prefix) + len(suffix) <= 2:
                    product[prefix + suffix] += a * b
        polynomial = {key: value for key, value in product.items() if value}
    return polynomial


def inverse_braid(word):
    return tuple(-x for x in reversed(word))


class ExactTripleMilnorTests(unittest.TestCase):
    def test_pair_linking_and_purity(self):
        self.assertEqual(pure_linking_numbers(3, C), {(1, 2): 0, (1, 3): 0, (2, 3): 0})
        self.assertEqual(pure_linking_numbers(3, (1, 1, 2, 2)),
                         {(1, 2): 1, (1, 3): 0, (2, 3): 1})
        with self.assertRaises(PureThreeBraidRequired):
            pure_linking_numbers(3, (1,))
        with self.assertRaises(ValueError):
            pure_linking_numbers(3, (0,))

    def test_mixed_magnus_matches_independent_polynomial(self):
        rng = random.Random(20261113)
        for _ in range(128):
            letters = [rng.choice((-1, 1)) * rng.randint(1, 3) for _ in range(12)]
            direct = direct_magnus_degree_two(letters)
            for i in (1, 2, 3):
                for j in (1, 2, 3):
                    if i != j:
                        self.assertEqual(mixed_magnus_coefficient(letters, i, j),
                                         direct.get((i, j), 0))
        with self.assertRaises(ValueError):
            mixed_magnus_coefficient((1,), 1, 1)

    def test_longitude_normalization_is_correct(self):
        images = artin_action(3, C)
        for i, image in enumerate(images, start=1):
            u = normalized_conjugator(image, i)
            self.assertEqual(sum((1 if x > 0 else -1) for x in u if abs(x) == i), 0)
            from nonabelian import free_reduce, inverse_free_word
            self.assertEqual(free_reduce(u + (i,) + inverse_free_word(u)), image)

    def test_commutator_inverse_alexander_blind_spot(self):
        m, inv = triple_milnor(3, C), triple_milnor(3, CI)
        self.assertEqual(m['mu123'], 1)
        self.assertEqual(inv['mu123'], -1)
        self.assertEqual(m['raw_cyclic_coefficients'], [1, 1, 1])
        self.assertEqual(inv['raw_cyclic_coefficients'], [-1, -1, -1])
        self.assertEqual(m['indeterminacy_gcd'], 0)
        self.assertEqual(triple_milnor(3, ())['mu123'], 0)
        self.assertEqual(normalized_alexander(3, C), normalized_alexander(3, CI))
        self.assertEqual(m['pairwise_linking'], inv['pairwise_linking'])

    def test_borromean_fixture_and_additivity(self):
        self.assertEqual(triple_milnor(3, (1, -2) * 3)['mu123'], 1)
        for integer in (-3, -2, -1, 0, 1, 2, 3):
            word = C * integer if integer >= 0 else CI * (-integer)
            self.assertEqual(triple_milnor_nilpotent(3, word)['mu123'], integer)
            if abs(integer) <= 2:
                self.assertEqual(triple_milnor(3, word)['mu123'], integer)

    def test_nilpotent_1000_commutators_exact(self):
        # Full Artin free words grow exponentially; cubic Magnus data have at
        # most 40 monomials per meridian regardless of braid word length.
        chain = C * 1000
        compact = triple_milnor_nilpotent(3, chain)
        self.assertEqual(compact['mu123'], 1000)
        self.assertLessEqual(max(compact['magnus_nonzero_terms']), 40)
        self.assertEqual(triple_milnor_nilpotent(3, CI*1000)['mu123'], -1000)
        self.assertNotEqual(triple_milnor_nilpotent(3, ())['mu123'], 1000)

    def test_cubic_crosscheck_random_zero_linking(self):
        rng = random.Random(412)
        for _ in range(36):
            word = ()
            for _ in range(rng.randint(0, 2)):
                base = C if rng.randrange(2) else CI
                word += base
            outcome = triple_milnor_nilpotent(3, word)
            self.assertEqual(outcome['mu123'], triple_milnor(3, word)['mu123'])

    def test_pure_conjugation_invariance(self):
        for conjugator in (A, B, A+B, (2, 1, 1, -2)):
            moved = conjugator + C + inverse_braid(conjugator)
            self.assertEqual(triple_milnor(3, moved)['mu123'], 1)
            moved_inverse = conjugator + CI + inverse_braid(conjugator)
            self.assertEqual(triple_milnor(3, moved_inverse)['mu123'], -1)

    def test_indeterminacy_nonzero_linking(self):
        word = (1, 1, 1, 1, 2, 2, 2, 2)
        original = triple_milnor(3, word)
        self.assertEqual(original['indeterminacy_gcd'], 2)
        self.assertEqual(original['status'], 'modulo_indeterminacy')
        for conjugator in (A, B, A+B):
            moved = triple_milnor(3, conjugator + word + inverse_braid(conjugator))
            self.assertEqual(moved['mu123'], original['mu123'])

    def test_seeded_pure_words_cyclic_consistency(self):
        rng = random.Random(20261114)
        generators = (A, B, (2, 1, 1, -2))
        for _ in range(200):
            word = ()
            for _ in range(rng.randint(1, 5)):
                gen = rng.choice(generators)
                word += gen if rng.randrange(2) else inverse_braid(gen)
            outcome = triple_milnor(3, word)
            g = outcome['indeterminacy_gcd']
            cyc = outcome['raw_cyclic_coefficients']
            self.assertEqual(len({a if not g else a % g for a in cyc}), 1)

    def test_phase_start_rebase_relabels_link(self):
        frames = synthetic_braid(3, C)
        rows = frames[:-1]
        for k in range(len(rows)):
            shifted = rows[k:] + rows[:k] + [rows[k]]
            braid = extract_closed_braid(shifted)
            info = token_order_milnor(shifted, braid.word)
            self.assertEqual(info['mu123_token_order'], 1)
        # A raw position-ordered number would give the wrong answer for a
        # phase shift that exchanges the x-ranks of two token identities.
        k = 2
        rotated = rows[k:] + rows[:k] + [rows[k]]
        info = token_order_milnor(rotated, extract_closed_braid(rotated).word)
        self.assertEqual(info['position_to_token_strands'], [1, 0, 2])
        self.assertEqual(info['mu123_position_order'], -1)
        self.assertEqual(info['mu123_token_order'], 1)

    def test_rmp_capture_emits_token_order_closure_invariant(self):
        a = analyze_cycle({'frames_by_visit': [synthetic_braid(3, C)]})[0]
        b = analyze_cycle({'frames_by_visit': [synthetic_braid(3, CI)]})[0]
        self.assertTrue(a['valid'] and b['valid'])
        self.assertEqual(a['pair_winding'], b['pair_winding'])
        self.assertEqual(a['alexander'], b['alexander'])
        self.assertEqual(a['milnor']['mu123_token_order'], 1)
        self.assertEqual(b['milnor']['mu123_token_order'], -1)
        comparison = compare_double_resolution(
            {'frames_by_visit': [synthetic_braid(3, C)]},
            {'frames_by_visit': [synthetic_braid(3, CI)]},
        )[0]
        self.assertFalse(comparison['closure_invariants_match'])
        self.assertFalse(comparison['milnor_match'])
        self.assertTrue(comparison['abelian_match'])

    def test_invalid_branches_and_size_budget(self):
        with self.assertRaises(PureThreeBraidRequired):
            triple_milnor(2, ())
        with self.assertRaises(PureThreeBraidRequired):
            triple_milnor(3, (1,))
        with self.assertRaises(ArtinWordBudgetExceeded):
            triple_milnor(3, C, max_letters=8)

if __name__ == '__main__':
    unittest.main()
