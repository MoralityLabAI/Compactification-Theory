import unittest
from itertools import product
from temporal_capacity import (minimum_heads,singleton_codes,tied_probe,tied_step,tied_trace,
    verify_tied,maximum_equidistant_binary_family,contracting_late_code)

class CapacityTests(unittest.TestCase):
    def test_tied_five_margin_one(self):
        self.assertEqual(minimum_heads(5,4,1),1)
        x=verify_tied(5,4,1)
        self.assertEqual((x['min_pair_distance'],x['max_pair_distance']),(1,2))

    def test_tied_five_margin_two(self):
        self.assertEqual(minimum_heads(5,4,2),2)
        x=verify_tied(5,4,2)
        self.assertEqual((x['min_pair_distance'],x['max_pair_distance']),(2,2))

    def test_margin_three_infeasible(self):
        self.assertIsNone(minimum_heads(5,4,3))
        self.assertFalse(verify_tied(5,4,3)['possible'])

    def test_many_n_and_visits(self):
        for n in range(5,22):
            for T in range(1,8):
                for margin in (1,2):
                    with self.subTest(n=n,T=T,margin=margin):
                        x=verify_tied(n,T,margin)
                        self.assertEqual(x['heads'],(n-1+(margin==2)+T-1)//T)

    def test_independent_four_bit_anticode_oracle(self):
        self.assertEqual([maximum_equidistant_binary_family(w) for w in (1,2,3,4)],[1,2,4,4])
        self.assertEqual(minimum_heads(5,1,2),5)

    def test_tied_step_uses_one_unchanged_probe(self):
        codes=singleton_codes(5,4,1)
        for label in range(5):
            state=(label,0);history=[]
            for _ in range(4):
                history.extend(tied_probe(state,codes,1));state=tied_step(state)
            self.assertEqual(tuple(history),tied_trace(label,codes,4,1))

    def test_contractive_late_innovation_stops(self):
        x=contracting_late_code([0.0,1.0,2.0,3.0],0.5,0.24,12)
        self.assertEqual(x['terminal_innovation_visit'],4)
        self.assertTrue(all(not any(code[t] for code in x['codes']) for t in range(4,12)))

if __name__=='__main__':unittest.main()

class CountOnlyBounds(unittest.TestCase):
    def test_multiset_capacity(self):
        from visit_phase_controls import unordered_qary_upper_bound
        self.assertEqual(unordered_qary_upper_bound(4,1),5)
        self.assertEqual(unordered_qary_upper_bound(4,2),35)
        self.assertEqual(2**(4*1),16)

    def test_k8_count_only_max_edges(self):
        from visit_phase_controls import clique_max_separated_with_categories
        self.assertEqual(clique_max_separated_with_categories(8,5),25)
        self.assertEqual(clique_max_separated_with_categories(8,8),28)
        self.assertEqual(clique_max_separated_with_categories(8,1),0)
