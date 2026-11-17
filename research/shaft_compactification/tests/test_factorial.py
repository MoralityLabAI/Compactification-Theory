import unittest
from research.shaft_compactification.factorial_auc_audit import mann_whitney
from research.shaft_compactification.factorial_reference import reference,energy


class FactorialTests(unittest.TestCase):
    def test_aucs(self):
        self.assertEqual(mann_whitney(1,2,2,0),1)
        self.assertEqual(mann_whitney(1,2,0,2),.5)
        self.assertEqual(mann_whitney(1,2,0,0),0)
    def test_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            mann_whitney(0,3,0,0)
        with self.assertRaises(ValueError):
            mann_whitney(1,1,2,0)
    def test_equal_rank_separates_direction(self):
        x=reference()
        self.assertEqual(x["both_readout_ranks"],1)
        self.assertEqual(x["contrasts"]["first_axis"]["learned_direction_energy"],1)
        self.assertEqual(x["contrasts"]["first_axis"]["native_direction_energy"],0)
        self.assertEqual(x["contrasts"]["second_axis"]["learned_direction_energy"],0)
        self.assertEqual(x["contrasts"]["second_axis"]["native_direction_energy"],1)
    def test_scaled_contrast(self):
        self.assertEqual(energy((1,0),(3,0)),9)
if __name__=="__main__":
    unittest.main()
