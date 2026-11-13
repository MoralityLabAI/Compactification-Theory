"""OIC-32 exhaustive exceptional prime-case regression tests."""
import unittest
from oic32_small import exhaustive_small_case,reference
from oic32_global import minimum_cross_shift_distance


class ExceptionalPrimeTests(unittest.TestCase):
    def test_all_registered_normalized_schedules(self):
        expected={(2,5):(16,2),(2,6):(32,3),(3,7):(729,2),(3,8):(2187,3)}
        for (p,horizon),(number,best) in expected.items():
            result=exhaustive_small_case(p,horizon)
            self.assertEqual(result['normalized_schedules_checked'],number)
            self.assertEqual(result['maximum_minimum_cross_shift_distance'],best)
            self.assertEqual(minimum_cross_shift_distance(p,tuple(result['first_optimal_word'])),best)

    def test_global_translation_preserves_all_cross_shift_distances(self):
        for p,word in ((2,(0,0,0,0,0,0)),(3,(0,0,0,1,1,1,2,2))):
            original=minimum_cross_shift_distance(p,word)
            for c in range(p):
                translated=tuple((m+c)%p for m in word)
                self.assertEqual(minimum_cross_shift_distance(p,translated),original)

    def test_truncated_shorter_horizon_failure(self):
        for p,h in ((2,5),(3,7)):
            short=exhaustive_small_case(p,h)
            self.assertLess(short['maximum_minimum_cross_shift_distance'],3)
            long=exhaustive_small_case(p,h+1)
            self.assertGreaterEqual(long['maximum_minimum_cross_shift_distance'],3)

    def test_receipt(self):
        r=reference()
        self.assertEqual([x['normalized_schedules_checked'] for x in r['cases']],[16,32,729,2187])
        self.assertTrue(r['cases'][1]['guarantees_one_substitution'])
        self.assertTrue(r['cases'][3]['guarantees_one_substitution'])

    def test_invalid_inputs(self):
        for p,h in ((4,6),(2,0),(3,9),(1,5)):
            with self.assertRaises(ValueError):
                exhaustive_small_case(p,h)

if __name__=='__main__':
    unittest.main()
