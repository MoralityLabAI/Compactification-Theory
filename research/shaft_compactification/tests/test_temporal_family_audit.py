import unittest
from research.shaft_compactification.temporal_family_audit import auc_from_counts

class TemporalFamilyAuditTests(unittest.TestCase):
    def test_auc_perfect(self):
        self.assertEqual(auc_from_counts(2,4,8,0),1)
    def test_auc_ties(self):
        self.assertEqual(auc_from_counts(2,4,0,8),.5)
    def test_auc_reverse(self):
        self.assertEqual(auc_from_counts(2,4,0,0),0)
    def test_auc_mixed(self):
        self.assertEqual(auc_from_counts(2,4,3,2),.5)
    def test_rejects_invalid(self):
        for args in ((0,3,0,0),(2,3,5,2),(2,3,-1,0)):
            with self.assertRaises(ValueError):
                auc_from_counts(*args)

if __name__=="__main__":
    unittest.main()
