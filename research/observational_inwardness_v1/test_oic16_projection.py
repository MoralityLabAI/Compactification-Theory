"""Exact finite projection tests under unknown initial phase, including faults."""
import json
import unittest
from pathlib import Path

from oic16_projection import (
    projected_steps, projected_marker, projected_sets, oracle_sets,
    trace, decode, panel, evaluate,
)

HERE = Path(__file__).resolve().parent


def records():
    items=[]
    for augment in ('clean','aug'):
        for chunk in ('s01','s23','s4'):
            items.extend(json.loads((HERE/'results'/f'{augment}_{chunk}.json').read_text())['runs'])
    return items


class OIC16ProjectionTests(unittest.TestCase):
    def test_all_learned_models_recover_generator_class(self):
        r=records()
        self.assertEqual(len(r),20)
        for item in r:
            shifts=projected_steps(item['weights'])
            self.assertEqual(shifts[0],0)
            self.assertIn(shifts[1],(-1,1))
            for n in range(4,13):
                self.assertEqual(projected_marker(item['weights'],n),0)

    def test_projected_codebooks_gauge_equivalent(self):
        for item in records():
            for n in range(4,13):
                for word in ((0,)*(2*n),(1,)*n,(1,)*n+(0,)*2):
                    self.assertEqual(projected_sets(n,word,item['weights']),oracle_sets(n,word))

    def test_projection_guarantees_raw_fault_correction(self):
        for item in records():
            for n in (4,6,8,10,12):
                word=(1,)*n+(0,)*2
                for i in range(n+3):
                    self.assertEqual(panel(n,word,item['weights'],flip_position=i)['accuracy'],1.)
                    self.assertEqual(panel(n,word,item['weights'],erase_position=i)['accuracy'],1.)
                self.assertEqual(panel(n,(0,)*(2*n),item['weights'])['accuracy'],.5)

    def test_one_step_oracle_via_explicit_law(self):
        for n in range(4,9):
            for f in (0,1):
                for z in range(n):
                    self.assertEqual(trace(n,f,z,(1,)*n,(0,1),0)[0],0)

    def test_input_validation_and_empty(self):
        w=records()[0]['weights']
        self.assertEqual(panel(4,tuple(),w)['accuracy'],.5)
        with self.assertRaises(ValueError):
            decode(4,(1,1),w,tuple([0,2,1]))
        with self.assertRaises(ValueError):
            projected_marker(w,1)

    def test_projection_receipt(self):
        j=json.loads((HERE/'results'/'oic16_summary.json').read_text())
        self.assertEqual(j['run_count'],20)
        self.assertTrue(j['all_projection_generators_correct'])
        self.assertTrue(j['all_even_fault_positions_correct_after_projection'])


if __name__=='__main__':
    unittest.main()
