import unittest
from research.shaft_compactification.task_auc_audit import auc_from_pair_counts

class TaskAUCAuditTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(auc_from_pair_counts(2,3,6,0),1.)
    def test_random_tie(self):
        self.assertEqual(auc_from_pair_counts(2,3,0,6),.5)
    def test_perfect_reverse(self):
        self.assertEqual(auc_from_pair_counts(2,3,0,0),0.)
    def test_rank_score_half_ties(self):
        self.assertEqual(auc_from_pair_counts(1,4,2,2),.75)
    def test_reject_malformed(self):
        with self.assertRaises(ValueError):
            auc_from_pair_counts(0,3,0,0)
        with self.assertRaises(ValueError):
            auc_from_pair_counts(2,3,5,2)
if __name__=="__main__":unittest.main()
