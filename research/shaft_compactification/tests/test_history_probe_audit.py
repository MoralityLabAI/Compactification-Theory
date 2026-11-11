import unittest
from research.shaft_compactification.history_probe_audit import mann_whitney,audited_rate
class HistoryProbeAuditTests(unittest.TestCase):
    def test_exact_auc_perfect(self):self.assertEqual(mann_whitney(2,3,6,0),1)
    def test_auc_equal(self):self.assertEqual(mann_whitney(2,3,0,6),.5)
    def test_auc_reverse(self):self.assertEqual(mann_whitney(2,3,0,0),0)
    def test_reject_overflow(self):
        with self.assertRaises(ValueError):mann_whitney(1,2,2,1)
    def test_reject_missing_class(self):
        with self.assertRaises(ValueError):mann_whitney(0,2,0,0)
    def test_rates(self):
        self.assertAlmostEqual(audited_rate(20,492),20/492)
        with self.assertRaises(ValueError):audited_rate(4,3)
if __name__=="__main__":unittest.main()
