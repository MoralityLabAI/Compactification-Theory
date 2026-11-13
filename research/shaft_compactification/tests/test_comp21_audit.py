import unittest
from research.shaft_compactification.comp21_audit import exact_auc, audit


class Comp21AuditTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(exact_auc(2,3,6,0),1.)
    def test_ties(self):
        self.assertEqual(exact_auc(2,3,0,6),.5)
    def test_reverse(self):
        self.assertEqual(exact_auc(2,3,0,0),0.)
    def test_reject_negative(self):
        with self.assertRaises(ValueError):
            exact_auc(1,2,-1,0)
    def test_reject_extra_wins(self):
        with self.assertRaises(ValueError):
            exact_auc(1,2,3,0)
    def test_reject_undefined(self):
        with self.assertRaises(ValueError):
            exact_auc(0,4,0,0)
    def test_pinned_results(self):
        r=audit()
        self.assertTrue(r["status"]=="verified")
        self.assertEqual(r["heldout_gate"],4)


if __name__=="__main__":
    unittest.main()
