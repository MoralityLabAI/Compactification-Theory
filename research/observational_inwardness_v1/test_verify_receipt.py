"""OIC-14 source-independent receipt tests, no PyTorch dependency."""
from __future__ import annotations
import copy
import json
import math
import unittest
from pathlib import Path

from verify_receipt import (SUMMARY, RECEIPT, exact_trace, inspect, verify, projected_exact)
from projection import detector_lattice, nearest_shift


class OIC14VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from verify_receipt import gzip, base64
        cls.source=json.loads(gzip.decompress(base64.b64decode(RECEIPT.read_text().strip())))

    def test_full_archive_and_code_pins(self):
        answer=verify()
        self.assertEqual(answer['manifest']['model_fits'],180)
        self.assertEqual(answer['manifest']['run_count'],90)
        self.assertEqual(answer['stats']['projected_correct_grid_fit_count'],60)
        self.assertEqual(answer['stats']['projected_correct_fit_count'],115)

    def test_no_missing_training_conditions(self):
        rows=self.source['runs']
        keys={(r['n'],r['architecture'],r['training_split'],r['seed']) for r in rows}
        self.assertEqual(len(keys),90)
        for n in (4,5,6):
            for a in ('orthogonal_local','orthogonal_grid','unconstrained'):
                for split in ('full','heldout'):
                    self.assertEqual(len([r for r in rows if (r['n'],r['architecture'],r['training_split'])==(n,a,split)]),5)

    def test_trace_tamper_is_detected(self):
        altered=copy.deepcopy(self.source)
        first=altered['runs'][0]['evaluations']['latch']['predicted']['fixed'][0]
        first[1]^=1
        with self.assertRaises(ValueError):inspect(altered)

    def test_fit_parameter_tamper_is_detected(self):
        altered=copy.deepcopy(self.source)
        for r in altered['runs']:
            if r['architecture']=='orthogonal_grid':
                r['training']['rotating']['learned_values']['omega'][0]=0.0
                break
        with self.assertRaises(ValueError):inspect(altered)

    def test_group_projection_is_equivariant_in_exact_finite_state(self):
        for n in range(2,20):
            for j in range(-3*n,3*n):
                q,delta=nearest_shift(n,2*math.pi*j/n)
                self.assertEqual(q,j%n)
                self.assertLess(delta,1e-12)
                for z in range(n):
                    for shift in range(n):
                        self.assertEqual(((z+shift)%n+q)%n,((z+q)%n+shift)%n)

    def test_group_projection_conditional_lattice_property(self):
        for n in (4,5,6):
            for q in (0,1):
                omega=2*math.pi*q/n+0.01
                selected,_=nearest_shift(n,omega)
                self.assertEqual(selected,q)
                self.assertEqual(detector_lattice(n,8.0,-6.0),(True,)+(False,)*(n-1))
                # On every phase and any horizon, exact modular transition with
                # the correct generator and readout equals the finite oracle.
                for z in range(n):
                    for t in range(3*n):
                        predicted=int((z+t*selected)%n==0)
                        expected=int((z+t*q)%n==0)
                        self.assertEqual(predicted,expected)

    def test_heldout_grid_matches_oracle_raw_distance(self):
        r=verify()['stats']['summary']
        for item in r:
            if item['architecture']=='orthogonal_grid' and item['training_split']=='heldout':
                self.assertEqual(item['whole_trace_accuracy_fixed'],1.0)
                self.assertEqual(item['whole_trace_accuracy_rotating'],1.0)
                self.assertEqual(item['latch_min_distances'],item['latch_oracle_min_distances'])

    def test_oracle_dynamics_is_independent_of_model_fit(self):
        self.assertEqual(exact_trace(4,'fixed',0,'1111'),[0,0,0,0,0])
        self.assertEqual(exact_trace(4,'fixed',1,'1111'),[0,1,0,1,0])
        self.assertEqual(exact_trace(4,'rotating',1,'1111'),[0,1,0,1,1])


if __name__=='__main__':unittest.main()
