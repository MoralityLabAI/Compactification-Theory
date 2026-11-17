"""Check that the checked-in synthetic reference has its declared proof scope."""
from fractions import Fraction
from pathlib import Path
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class ReferenceReceiptTests(unittest.TestCase):
    def test_three_configs_and_all_visits(self):
        receipt=json.loads((ROOT/'results/scm09_cpu_reference.json').read_text())
        self.assertEqual(receipt['schema'],'scm09_exact_real_centered_interval_reference_v1')
        self.assertEqual(len(receipt['cases']),3)
        self.assertEqual([(c['dimension'],c['sequence'],c['tied'],c['reinjection'])
                        for c in receipt['cases']],[(8,8,True,False),(8,8,False,True),(16,12,True,False)])
        for case in receipt['cases']:
            self.assertEqual(case['null_braid_ideal_real_visits'],list(range(1,9)))
            self.assertLess(Fraction(case['validated_centered_post_max']),Fraction(1,10**12))
            self.assertGreater(Fraction(case['error_radius_improvement_lower']),10**12)
            self.assertFalse(case['float_execution_certified'])
            self.assertFalse(case['observed_link_certified'])

if __name__=='__main__':
    unittest.main()
