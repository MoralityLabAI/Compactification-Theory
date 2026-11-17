import unittest
from research.shaft_compactification.comp19_audit import mann_whitney

class Comp19AUCTests(unittest.TestCase):
    def test_all_wins(self):
        self.assertEqual(mann_whitney(2,3,6,0),1)
    def test_all_losses(self):
        self.assertEqual(mann_whitney(2,3,0,0),0)
    def test_all_ties(self):
        self.assertEqual(mann_whitney(2,3,0,6),.5)
    def test_partial_ties(self):
        self.assertEqual(mann_whitney(1,4,2,2),.75)
    def test_invalid_counts(self):
        for args in ((0,3,0,0),(2,3,7,0),(2,3,0,-1)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    mann_whitney(*args)

if __name__ == "__main__":
    unittest.main()
