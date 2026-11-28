import unittest
from research.shaft_compactification.finite_history_vocab import one,two,reference

class HistoryVocabularyTests(unittest.TestCase):
    def test_witness_each_vocab(self):
        for v in (4,6):
            self.assertEqual(one(1,0,v),one(2,1,v))
            self.assertNotEqual(two(1,0,v),two(2,1,v))
    def test_immediate_classes(self):
        r=reference()
        for v in (4,6):
            self.assertEqual(r['results'][str(v)]['immediate_signature_classes'],2**v)
    def test_future_refines(self):
        r=reference()
        self.assertTrue(all(x['strict_refinement'] for x in r['results'].values()))
    def test_first_query_signature_lengths(self):
        for v in (4,6):
            self.assertEqual(len(one(1,0,v)),v)
            self.assertEqual(len(two(1,0,v)),v*v)

if __name__=='__main__':unittest.main()
