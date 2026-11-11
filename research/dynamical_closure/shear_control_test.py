"""CPU numerical-feature controls; exact risks use shear_exact tests too."""
import unittest

import numpy as np

from shear_controls import allocate, curve, feature_banks, quantize, random_basis


class ShearControlsTests(unittest.TestCase):
    def test_budget_and_nested_quantization(self):
        raw, engineered = feature_banks()
        self.assertEqual(raw.shape, (961,4))
        self.assertEqual(engineered.shape, (961,4))
        self.assertEqual(allocate(10), (3,3,2,2))
        for bank, basis in ((raw, np.eye(4)), (engineered, random_basis(4))):
            previous = None
            for budget in range(11):
                code = quantize(bank, budget, basis)
                self.assertLessEqual(len(set(code)), 2**budget)
                if previous is not None:
                    mapping = {}
                    for fine, coarse in zip(code, previous):
                        self.assertEqual(mapping.setdefault(fine, coarse), coarse)
                previous = code

    def test_curve_exact_risk_and_random_basis(self):
        raw, engineered = feature_banks()
        q = random_basis(11)
        self.assertTrue(np.allclose(q.T @ q, np.eye(4), atol=1e-12))
        values = curve(engineered, q)
        self.assertEqual(values[0]['present_risk'], '30/31')
        self.assertEqual(values[0]['after_shear_risk'], '30/31')
        self.assertEqual([row['bits'] for row in values], list(range(11)))
        self.assertEqual([row['codes'] for row in values],
                         sorted(row['codes'] for row in values))


if __name__ == '__main__':
    unittest.main()
