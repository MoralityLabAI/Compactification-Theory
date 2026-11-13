import unittest
from research.shaft_compactification.comp22_audit import verify

class Comp22AuditTests(unittest.TestCase):
    def test_frozen_result_reconstructs(self):
        r=verify()
        self.assertEqual(r['model_count'],4)
        self.assertEqual(r['primary_exact_difference'],.072754)
    def test_confirmed_native_bit_accuracies(self):
        r=verify()
        self.assertGreater(r['mean_native_bit_accuracy_depth40']['canonical'],
                           r['mean_native_bit_accuracy_depth40']['direct'])
    def test_effective_probe_is_distinct_from_canonical(self):
        r=verify()
        self.assertGreater(r['mean_native_bit_accuracy_depth40']['canonical'],
                           r['mean_native_bit_accuracy_depth40']['behavioral'])
if __name__=='__main__':unittest.main()
