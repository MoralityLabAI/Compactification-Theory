import unittest
from research.shaft_compactification.comp20_audit import auc

class Comp20AUCTests(unittest.TestCase):
    def test_strict(self):
        self.assertEqual(auc(2,3,6,0),1)
        self.assertEqual(auc(2,3,0,0),0)
    def test_ties(self):
        self.assertEqual(auc(2,3,0,6),.5)
    def test_reject(self):
        with self.assertRaises(ValueError):auc(0,3,0,0)
        with self.assertRaises(ValueError):auc(2,3,6,1)
if __name__=="__main__":unittest.main()
