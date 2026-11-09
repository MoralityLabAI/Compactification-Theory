"""Grid-search decisions are exact-rational and solver-independent on replay."""
import copy
import json
import sys
from fractions import Fraction
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_certificate import select, verify_selection


class AdaptiveCertificateTests(unittest.TestCase):
    def test_first_pass_exact_threshold(self):
        receipt = select(Fraction('0.001'), grid=(30, 38, 54, 70), cutoff=1024)
        self.assertEqual(receipt['selection']['first_passing_M'], 70)
        self.assertEqual([c['target_passed'] for c in receipt['cases']],
                         [False, False, False, True])
        self.assertTrue(verify_selection(receipt))

    def test_no_threshold_pass_scans_entire_grid(self):
        receipt = select(Fraction('0.00001'), grid=(2, 6), cutoff=128)
        self.assertIsNone(receipt['selection']['first_passing_M'])
        self.assertEqual(len(receipt['cases']), 2)
        self.assertTrue(verify_selection(receipt))

    def test_replay_and_mutations(self):
        receipt = select(Fraction('0.02'), grid=(6, 10, 14), cutoff=128)
        self.assertEqual(receipt['selection']['first_passing_M'], 14)
        replay = json.loads(json.dumps(receipt))
        self.assertTrue(verify_selection(replay))
        replay['cases'][-1]['target_passed'] = False
        with self.assertRaises(AssertionError):
            verify_selection(replay)
        replay = copy.deepcopy(receipt)
        replay['cases'][0]['witness_vector_integer_pairs'][0][0][0] += 10000
        with self.assertRaises(AssertionError):
            verify_selection(replay)
        replay = copy.deepcopy(receipt)
        replay['cases'].pop(0)
        with self.assertRaises(AssertionError):
            verify_selection(replay)

    def test_input_validation(self):
        for grid in ((), (2, 2), (6, 2), (3, 4)):
            with self.assertRaises(ValueError):
                select(Fraction('0.1'), grid=grid)
        with self.assertRaises(ValueError):
            select(Fraction(0))


if __name__ == '__main__':
    unittest.main()
