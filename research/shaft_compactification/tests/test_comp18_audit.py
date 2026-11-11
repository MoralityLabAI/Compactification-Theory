import unittest
from research.shaft_compactification.comp18_audit import auc_from_counts
from research.shaft_compactification.finite_history_vocab import one,two,reference

class Comp18AuditTests(unittest.TestCase):
    def test_perfect_auc(self):
        self.assertEqual(auc_from_counts({"positives":2,"negatives":3,"strict_wins":6,"ties":0}),1.)
    def test_half_tie(self):
        self.assertEqual(auc_from_counts({"positives":2,"negatives":3,"strict_wins":0,"ties":6}),.5)
    def test_invalid_pairs_rejected(self):
        with self.assertRaises(ValueError):
            auc_from_counts({"positives":1,"negatives":1,"strict_wins":2,"ties":0})
        with self.assertRaises(ValueError):
            auc_from_counts({"positives":0,"negatives":2,"strict_wins":0,"ties":0})
    def test_finite_vocabulary_aliases(self):
        for v in (4,6):
            self.assertEqual(one(1,0,v),one(2,1,v))
            self.assertNotEqual(two(1,0,v),two(2,1,v))
    def test_exact_partition_sizes(self):
        r=reference()["results"]
        self.assertEqual(r["4"]["joint_one_and_two_step_classes"],54)
        self.assertEqual(r["6"]["joint_one_and_two_step_classes"],348)

if __name__=="__main__":unittest.main()
