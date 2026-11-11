import unittest
from research.shaft_compactification.rank_auc_reference import auc_from_bins

class RankAUCTests(unittest.TestCase):
    def test_perfect_ordering(self):
        self.assertEqual(auc_from_bins({"0":{"positive":0,"negative":2},"1":{"positive":3,"negative":0}}),1.)
    def test_reversed_ordering(self):
        self.assertEqual(auc_from_bins({"0":{"positive":4,"negative":0},"1":{"positive":0,"negative":6}}),0.)
    def test_ties(self):
        self.assertEqual(auc_from_bins({"0":{"positive":2,"negative":3}}),.5)
    def test_undefined_one_class(self):
        self.assertIsNone(auc_from_bins({"0":{"positive":0,"negative":3}}))
    def test_mixed_auc(self):
        self.assertEqual(auc_from_bins({"0":{"positive":1,"negative":1},"1":{"positive":1,"negative":1}}),.5)

if __name__ == "__main__":
    unittest.main()
