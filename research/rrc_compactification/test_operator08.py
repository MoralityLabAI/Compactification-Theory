"""RRC-08: independent finite algebra, honest inference and split gates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from . import operator08 as op
    from . import bench08
    from .rrc import euler_exponents,gap_coefficients
    from .rrc05 import (Program05,product_coefficients,log_derivative_coefficients,
                        proper_divisors,coefficient_residual)
except ImportError:
    import operator08 as op
    import bench08
    from rrc import euler_exponents,gap_coefficients
    from rrc05 import (Program05,product_coefficients,log_derivative_coefficients,
                       proper_divisors,coefficient_residual)

HERE=Path(__file__).resolve().parent


def learned():
    ckpt=json.loads((HERE/'results'/'rrc08_operator_weights.json').read_text())
    weights=ckpt['decoded_coefficients']
    if weights is None:raise AssertionError('operator was not compiled')
    return op.Operator08(tuple(weights[:4]),tuple(weights[4:]))


class ExactRecurrence(unittest.TestCase):
    def test_independent_small_partition_enumeration(self):
        # Direct recursive partitions over a finite allowed-part set. This is
        # independent of the coefficient polynomial DP used to make rosters.
        def brute(n,parts,cap):
            def ways(left,index):
                if index==len(parts):return int(left==0)
                step=parts[index]
                maximum=left//step if cap==0 else min(cap,left//step)
                return sum(ways(left-k*step,index+1) for k in range(maximum+1))
            return ways(n,0)
        for cap in (0,1,2):
            program=Program05(cap,5,(1,4))
            f=product_coefficients(12,program)
            parts=[p for p in range(1,13) if p%5 in (1,4)]
            self.assertEqual(f,[brute(n,parts,cap) for n in range(13)])
            b,a=op.serial_circuit(f,learned())
            self.assertEqual(a,euler_exponents(f))
            self.assertEqual(b,log_derivative_coefficients(f))

    def test_rr_reference_from_independent_oracle(self):
        circuit=learned()
        for minimum,residues in ((1,(1,4)),(2,(2,3))):
            f=gap_coefficients(op.PREFIX,minimum_part=minimum)
            b,a=op.serial_circuit(f,circuit)
            self.assertEqual(b,log_derivative_coefficients(f))
            self.assertEqual(a,euler_exponents(f))
            self.assertIsNone(op.coefficient_first_fault(f,a))
            self.assertEqual(op.propose_program(f,a),Program05(0,5,residues))

    def test_general_integer_series_and_negative_coefficients(self):
        circuit=learned()
        src=product_coefficients(op.PREFIX,Program05(0,5,(1,4)))
        for f in (src,op._reciprocal_series(src),
                  [v*(-1)**n for n,v in enumerate(src)],
                  op._mul_series(src,src,degree=op.PREFIX),
                  [1,2,-3,4,0,1,0]+[0]*(op.PREFIX-6)):
            b,a=op.serial_circuit(f,circuit)
            self.assertEqual(a,euler_exponents(f))
            self.assertEqual(b,log_derivative_coefficients(f))
            self.assertIsNone(op.coefficient_first_fault(f,a))

    def test_mismatched_operator_rejected_by_exact_verifier(self):
        true=learned()
        f=product_coefficients(op.PREFIX,Program05(0,5,(1,4)))
        a=op.serial_circuit(f,true)[1]
        a[11]+=1
        self.assertEqual(op.coefficient_first_fault(f,a),11)
        self.assertIsNone(op.propose_program(f,a))
        bad=op.Operator08((0,-1,0,0),(1,-1,0,0))
        trace=op.serial_circuit(f,bad)
        self.assertTrue(trace is None or op.coefficient_first_fault(f,trace[1]) is not None)

    def test_nonintegral_operator_fails_closed(self):
        # b[n]=nf[n], a[n]=b[n]/n + a[n-1] at n>=2 can be integral,
        # but b[n]=1 gives a[2]=1/2, which must be rejected.
        malformed=op.Operator08((0,0,0,1),(1,0,0,0))
        self.assertIsNone(op.serial_circuit([1,0,0,0],malformed))

    def test_validation_and_term_budget(self):
        with self.assertRaises(ValueError):op.Operator08((1,-1),(1,-1,0,0))
        with self.assertRaises(ValueError):op.Operator08((3,-1,0,0),(1,-1,0,0))
        with self.assertRaises(ValueError):op.serial_circuit([0,1],learned())
        with self.assertRaises(ValueError):op.serial_circuit([1,1.5],learned())
        with self.assertRaises(ValueError):op.parallel_circuit([1,1],learned(),-1)
        self.assertEqual(learned().syntax_bits,24)

    def test_serial_oracle_not_reachable_during_inference(self):
        f=product_coefficients(40,Program05(1,5,(1,3)))
        with mock.patch.object(op,'euler_exponents',side_effect=AssertionError('teacher leaked')):
            with mock.patch.object(op,'log_derivative_coefficients',side_effect=AssertionError('teacher leaked')):
                b,a=op.serial_circuit(f,learned())
        self.assertEqual(a,euler_exponents(f))
        self.assertEqual(b,log_derivative_coefficients(f))

    def test_parallel_depth_contract_and_serial_compatibility(self):
        c=learned()
        f=product_coefficients(32,Program05(0,5,(1,4)))
        exact=euler_exponents(f)
        early=op.parallel_circuit(f,c,2)[1]
        self.assertNotEqual(early,exact)
        late=op.parallel_circuit(f,c,33)[1]
        self.assertEqual(late,exact)
        self.assertEqual(op.serial_circuit(f,c)[1],late)

    def test_cached_depth_results_match_independent_computation(self):
        circuit=learned()
        f=product_coefficients(24,Program05(1,7,(1,3,5)))
        depths=(0,1,2,4,8,16,25)
        cache=op.parallel_snapshots(f,circuit,depths)
        for depth in depths:
            self.assertEqual(cache[depth],op.parallel_circuit(f,circuit,depth))
        with self.assertRaises(ValueError):
            op.parallel_snapshots(f,circuit,(-1,))

    def test_train_features_retain_no_future_counts(self):
        cases=op.roster(include_test=False,train_size=6,dev_size=4)
        self.assertEqual(len(cases['train']),6)
        self.assertEqual(sum(len(v) for k,v in cases.items() if k not in ('train','dev')),0)
        xb,yb,xa,ya=op.sample_features(cases['train'])
        self.assertEqual(len(xb),6*16)
        self.assertTrue(all(len(x)==4 for x in xb+xa))
        self.assertTrue(all(len(c.prefix)==op.PREFIX+1 for c in cases['train']))

    def test_unseen_family_and_collision_controls(self):
        cases=op.roster(include_test=True,train_size=10,dev_size=8,unseen_per_period=2)
        self.assertEqual(len(cases['power']),12)
        self.assertEqual(len(cases['inverse']),12)
        self.assertEqual(len(cases['signed_twist']),12)
        self.assertEqual(len(cases['collision']),12)
        self.assertEqual(len(cases['gap']),20)
        train={c.prefix for c in cases['train']}
        self.assertTrue(train.isdisjoint({c.prefix for c in cases['unseen_period']}))
        for left,right in zip(cases['collision'][::2],cases['collision'][1::2]):
            self.assertEqual(left.prefix,right.prefix)
            self.assertNotEqual(left.full,right.full)
            self.assertEqual(op.serial_circuit(left.prefix,learned()),
                             op.serial_circuit(right.prefix,learned()))

    def test_mutant_future_false_acceptance_is_charged(self):
        data=op.roster(include_test=True,train_size=10,dev_size=8,unseen_per_period=2)
        for base,mutant in zip(data['collision'][::2],data['collision'][1::2]):
            r0=op.score(base,learned())
            r1=op.score(mutant,learned())
            self.assertEqual(r0['accepted_program'],r1['accepted_program'])
            self.assertEqual(r0['correct_future']+r1['correct_future'],1)
            self.assertEqual(r0['false_accept']+r1['false_accept'],1)

    def test_exact_result_abstains_on_irregular_part_set(self):
        f=op.arbitrary_part_coefficients(op.PREFIX,[1,4,9,16,25],cap=0)
        b,a=op.serial_circuit(f,learned())
        self.assertEqual(a,euler_exponents(f))
        self.assertIsNone(op.propose_program(f,a))


class CheckpointsAndReplay(unittest.TestCase):
    def test_training_design_identified_and_snapped(self):
        operator=bench08.weights()['operator'][0]
        self.assertEqual(operator['training_examples'],192)
        self.assertEqual(operator['degree_supervised'],16)
        self.assertEqual(operator['training_rows'],3072)
        self.assertEqual(operator['b_fit']['design_rank'],4)
        self.assertEqual(operator['a_fit']['design_rank'],4)
        self.assertLess(operator['snap_max_error'],1e-6)
        self.assertEqual(operator['decoded_coefficients'],[1,-1,0,0,1,-1,0,0])
        self.assertFalse(operator['inference_teacher_access'])
        self.assertTrue(operator['training_teacher_traces'])

    def test_negative_controls_do_not_compile(self):
        models=bench08.weights()
        self.assertFalse(models['local'][0]['executable'])
        self.assertFalse(models['shuffled'][0]['executable'])

    def test_prefix_robustness_and_batched_evaluation(self):
        c=op.roster(include_test=True,train_size=12,dev_size=8,unseen_per_period=2)['unseen_period'][0]
        good=op.score(c,learned())
        control=op.score_trace(c,bench08.feedforward(c,bench08.weights()['local'][0]))
        self.assertEqual(good['verified_trace'],1)
        self.assertEqual(good['accepted_program'],1)
        self.assertEqual(good['correct_future'],1)
        self.assertEqual(control['verified_trace'],0)

    def test_one_family_internal_receipt_and_source_sha(self):
        c=op.roster(include_test=True,train_size=12,dev_size=8,unseen_per_period=2)['gap'][1]
        r=op.score(c,learned())
        self.assertTrue(r['verified_trace'])
        self.assertEqual(r['exact_coordinates'],op.PREFIX)
        self.assertIn('operator08.py',bench08.source_hashes())
        self.assertEqual(bench08.weights()['operator'][0]['training_data_sha256'],
                         json.loads((HERE/'results'/'rrc08_operator_weights.json').read_text())['training_data_sha256'])


if __name__=='__main__':unittest.main()
