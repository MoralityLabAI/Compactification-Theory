"""Exact nonabelian regression fixtures for winding-invisible pure braids."""
from __future__ import annotations

import pathlib
import random
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from knot import extract_closed_braid, normalized_alexander, synthetic_braid
from nonabelian import (ArtinWordBudgetExceeded, artin_action,
                       artin_witness, free_reduce, inverse_free_word)
from winding import verify_pure_braid
from rmp_capture import analyze_cycle, compare_double_resolution, sha256_tensor


class ArtinActionTests(unittest.TestCase):
    def test_free_group_cancellation(self):
        self.assertEqual(free_reduce([1, 2, -2, -1, 3, -3, 2]), (2,))
        self.assertEqual(inverse_free_word((1, -2, 3)), (-3, 2, -1))
        with self.assertRaisesRegex(ValueError, 'zero'):
            free_reduce([0])

    def test_braid_relations_and_inverse(self):
        self.assertEqual(artin_action(3, [1, 2, 1]), artin_action(3, [2, 1, 2]))
        self.assertEqual(artin_action(4, [1, 3]), artin_action(4, [3, 1]))
        for n in (2, 3, 4):
            for i in range(1, n):
                self.assertTrue(artin_witness(n, [i, -i])['is_identity'])
                self.assertTrue(artin_witness(n, [-i, i])['is_identity'])
                self.assertFalse(artin_witness(n, [i])['is_identity'])

    def test_random_word_inverse_is_identity(self):
        rng = random.Random(20261113)
        for n in (3, 4):
            for _ in range(32):
                word = [rng.choice((-1, 1))*rng.randint(1, n-1) for _ in range(5)]
                inverse = [-g for g in reversed(word)]
                self.assertTrue(artin_witness(n, word+inverse)['is_identity'])

    def test_commutator_is_invisible_to_pairwise_winding(self):
        a = [1, 1]
        b = [2, 2]
        commutator = a+b+[-x for x in reversed(a)]+[-x for x in reversed(b)]
        inverse = [-x for x in reversed(commutator)]
        for word in (commutator, inverse):
            frames = synthetic_braid(3, word)
            extraction = extract_closed_braid(frames)
            self.assertEqual(extraction.word, tuple(word))
            self.assertEqual(verify_pure_braid(frames, extraction),
                             {(0,1):0, (0,2):0, (1,2):0})
            self.assertFalse(artin_witness(3,word)['is_identity'])
        self.assertTrue(artin_witness(3,[])['is_identity'])
        self.assertNotEqual(artin_action(3,commutator), artin_action(3,inverse))
        self.assertEqual(normalized_alexander(3,commutator),
                         normalized_alexander(3,inverse))

    def test_conjugacy_not_automorphism_equality(self):
        # Markov-conjugate braids close to the same link, but have different
        # based Artin actions. Never call the image a link invariant.
        a = [1,1,2,2,-1,-1,-2,-2]
        b = [2]+a+[-2]
        self.assertNotEqual(artin_action(3,a), artin_action(3,b))
        self.assertEqual(normalized_alexander(3,a), normalized_alexander(3,b))

    def test_analyzer_detects_abelian_and_alexander_blind_spot(self):
        word = [1,1,2,2,-1,-1,-2,-2]
        inverse = [-x for x in reversed(word)]
        a_trace = {'frames_by_visit':[synthetic_braid(3,word)]}
        b_trace = {'frames_by_visit':[synthetic_braid(3,inverse)]}
        a,b=analyze_cycle(a_trace)[0],analyze_cycle(b_trace)[0]
        self.assertTrue(a['valid'] and b['valid'])
        self.assertEqual(a['pair_winding'],b['pair_winding'])
        self.assertEqual(a['alexander'],b['alexander'])
        self.assertEqual(a['canonical_seifert_genus'],b['canonical_seifert_genus'])
        self.assertNotEqual(a['artin']['images'],b['artin']['images'])
        self.assertTrue(a['artin']['nonidentity'] and b['artin']['nonidentity'])
        result=compare_double_resolution(a_trace,b_trace)[0]
        self.assertTrue(result['abelian_match'])
        self.assertFalse(result['nonabelian_match'])
        self.assertFalse(result['match'])
        self.assertEqual(result['nonabelian_comparison'],'exact_images')

    def test_budget_rejection_remains_explicit(self):
        trace={'frames_by_visit':[synthetic_braid(3,[1,1,2,2,-1,-1,-2,-2])]}
        record=analyze_cycle(trace,max_artin_letters=8)[0]
        self.assertTrue(record['valid'])
        self.assertEqual(record['artin']['status'],'budget_exceeded')
        self.assertFalse(compare_double_resolution(trace,trace,max_artin_letters=8)[0]['match'])

    def test_sixteen_winding_invisible_commutators(self):
        for a in (-2,-1,1,2):
            for b in (-2,-1,1,2):
                x=[1 if a>0 else -1]*(2*abs(a))
                y=[2 if b>0 else -2]*(2*abs(b))
                word=x+y+[-k for k in reversed(x)]+[-k for k in reversed(y)]
                frames=synthetic_braid(3,word)
                observed=extract_closed_braid(frames)
                self.assertEqual(verify_pure_braid(frames,observed),
                                 {(0,1):0,(0,2):0,(1,2):0})
                self.assertFalse(artin_witness(3,observed.word)['is_identity'])

    def test_hashing_does_not_require_numpy(self):
        t = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
        expected = sha256_tensor(t)
        with patch.object(torch.Tensor, 'numpy', side_effect=RuntimeError('NumPy unavailable')):
            self.assertEqual(sha256_tensor(t), expected)

    def test_size_budget_fails_closed(self):
        with self.assertRaises(ArtinWordBudgetExceeded):
            artin_action(3,[1,1,2,2,-1,-1,-2,-2],max_letters=8)


if __name__ == '__main__':
    unittest.main()
