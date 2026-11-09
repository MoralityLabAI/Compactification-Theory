"""RRC-06 exact recurrence, raw-information boundary and frozen replay tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

try:
    from .rrc06 import (PREFIX,HORIZON,SEED,SPLIT_SIZES,Program05,roster,raw_features,
                        teacher,b_series_residual,certify,rounded_predictions,
                        first_nonzero,score_case)
    from .rrc05 import product_coefficients,coefficient_residual
    from .raw_student06 import (infer_raw,parameter_count,portable_forward,
                                RECURRENT_HIDDEN,FEEDFORWARD_HIDDEN)
    from .bench06 import run,load_checkpoint
except ImportError:
    from rrc06 import (PREFIX,HORIZON,SEED,SPLIT_SIZES,Program05,roster,raw_features,
                       teacher,b_series_residual,certify,rounded_predictions,
                       first_nonzero,score_case)
    from rrc05 import product_coefficients,coefficient_residual
    from raw_student06 import (infer_raw,parameter_count,portable_forward,
                               RECURRENT_HIDDEN,FEEDFORWARD_HIDDEN)
    from bench06 import run,load_checkpoint

HERE=Path(__file__).resolve().parent


class ExactEulerBoundaryTests(unittest.TestCase):
    def test_direct_series_oracle_and_integer_trace(self):
        for cap in (0,1,2):
            f=product_coefficients(PREFIX,Program05(cap,7,(1,3,5)))
            a,b=teacher(f)
            self.assertEqual(coefficient_residual(f,a),[0]*(PREFIX+1))
            self.assertEqual(b_series_residual(f,b),[0]*(PREFIX+1))
            result=certify(f,a,b)
            self.assertTrue(result['verified_joint_trace'])
            self.assertIsNotNone(result['program'])

    def test_proposed_a_first_fault_and_exact_magnitude(self):
        f=product_coefficients(PREFIX,Program05(2,11,(1,4,9)))
        a,b=teacher(f)
        for n,delta in ((1,1),(7,-2),(31,1),(64,3)):
            modified=a.copy();modified[n]+=delta
            err=coefficient_residual(f,modified)
            self.assertEqual(first_nonzero(err),n)
            self.assertEqual(err[:n],[0]*n)
            self.assertEqual(err[n],-n*delta)
            cert=certify(f,modified,b)
            self.assertIsNone(cert['program'])
            self.assertEqual(cert['first_exponent_series_residual'],n)

    def test_proposed_b_first_fault_and_exact_magnitude(self):
        f=product_coefficients(PREFIX,Program05(0,13,(0,5,9)))
        a,b=teacher(f)
        for n,delta in ((1,1),(3,-2),(30,3),(64,1)):
            altered=b.copy();altered[n]+=delta
            r=b_series_residual(f,altered)
            self.assertEqual(first_nonzero(r),n)
            self.assertEqual(r[n],-delta)
            cert=certify(f,a,altered)
            self.assertTrue(cert['verified_exponent_trace'])
            self.assertFalse(cert['verified_joint_trace'])

    def test_rogers_ramanujan_controls(self):
        from pathlib import Path
        for minimum,residues in ((1,(1,4)),(2,(2,3))):
            try:from .rrc import gap_coefficients
            except ImportError:from rrc import gap_coefficients
            f=gap_coefficients(PREFIX,minimum_part=minimum)
            a,b=teacher(f)
            cert=certify(f,a,b)
            self.assertTrue(cert['verified_joint_trace'])
            self.assertEqual(cert['program'],Program05(0,5,residues))

    def test_reject_noninteger_and_nonfinite_predictions(self):
        f=product_coefficients(PREFIX,Program05(0,2,(1,)))
        a,b=teacher(f)
        with self.assertRaises(ValueError):certify(f,a,b[:-1])
        with self.assertRaises(ValueError):certify(f,[0.0]+a[1:],b)
        with self.assertRaises(ValueError):raw_features([1,2,3])
        with self.assertRaises(ValueError):rounded_predictions([[float('nan')]*4]*(PREFIX+1))
        with self.assertRaises(ValueError):b_series_residual([2,1],[0,1])

    def test_decode_uses_only_logits_and_finite_values(self):
        logits=[[0.,1.,0.,0.]]*(PREFIX+1)
        a,b=rounded_predictions(logits)
        self.assertEqual(a,[0]*(PREFIX+1))
        self.assertEqual(b,[0]*(PREFIX+1))
        x=[row.copy() for row in logits]
        x[27]=[2.,0.,0.,0.]
        self.assertEqual(rounded_predictions(x)[0][27],-1)


class FrozenRosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.parts=roster()

    def test_exact_sizes_unique_product_prefixes_and_no_test_training(self):
        parts=self.parts
        self.assertEqual({k:len(v) for k,v in parts.items()},SPLIT_SIZES)
        observed=[]
        for split in ('train','dev','unseen_period','unseen_cap'):
            observed.extend(c.prefix for c in parts[split])
        self.assertEqual(len(observed),len(set(observed)))
        private=roster(include_test=False)
        self.assertTrue(all(not private[k] for k in
                            ('unseen_period','unseen_cap','local_gap','nonperiodic',
                             'out_of_grammar','collision')))

    def test_collision_prefixes_and_future_differences(self):
        for i in range(5):
            left,right=self.parts['collision'][2*i:2*i+2]
            self.assertEqual(left.prefix,right.prefix)
            self.assertEqual(raw_features(left.prefix),raw_features(right.prefix))
            self.assertNotEqual(left.full,right.full)
            self.assertEqual(next(n for n in range(65,129)
                                  if left.full[n]!=right.full[n]),
                             (67,79,101,113,127)[i])

    def test_source_family_and_new_semantic_cap_isolation(self):
        self.assertTrue(all('periodic_product'==c.family for c in self.parts['train']))
        self.assertTrue(all('cap=2' in c.source for c in self.parts['unseen_cap']))
        self.assertTrue(all('cap=2' not in c.source for c in self.parts['train']))
        self.assertTrue(all(c.family=='nonperiodic' for c in self.parts['nonperiodic']))

    def test_raw_inference_does_not_call_algebra_teacher(self):
        ckpt=load_checkpoint('recurrent')
        prefix=self.parts['unseen_period'][0].prefix
        with patch.object(sys.modules[raw_features.__module__], 'teacher',
                          side_effect=AssertionError('inference reached exact teacher')):
            output=infer_raw(prefix,ckpt)
        self.assertEqual(len(output),PREFIX+1)
        self.assertTrue(all(len(item)==4 for item in output))

    def test_weight_count_and_zero_loop_ablation(self):
        self.assertEqual(parameter_count('recurrent'),2028)
        self.assertEqual(parameter_count('feedforward'),2084)
        self.assertEqual(RECURRENT_HIDDEN,23)
        self.assertEqual(FEEDFORWARD_HIDDEN,4)
        a,b=self.parts['unseen_period'][:2]
        ckpt=load_checkpoint('recurrent')
        self.assertEqual(infer_raw(a.prefix,ckpt,loops=0),
                         infer_raw(b.prefix,ckpt,loops=0))
        with self.assertRaises(ValueError):infer_raw(a.prefix,ckpt,loops=9)
        with self.assertRaises(ValueError):infer_raw(a.prefix,load_checkpoint('feedforward'),loops=2)

    def test_frozen_receipt_and_failure_gates(self):
        result=run()
        frozen=json.loads((HERE/'results'/'rrc06_pilot.json').read_text())
        self.assertEqual(result,frozen)
        self.assertEqual(result['metrics']['exact_symbolic']['unseen_period']['accepted_programs'],72)
        self.assertEqual(result['metrics']['exact_symbolic']['unseen_cap']['accepted_programs'],36)
        self.assertEqual(result['metrics']['exact_symbolic']['collision']['false_accepts'],5)
        self.assertEqual(result['metrics']['recurrent_2']['unseen_period']['accepted_programs'],0)
        self.assertEqual(result['metrics']['feedforward']['unseen_period']['accepted_programs'],0)
        self.assertEqual(result['recurrent_unseen_period_depth']['1']['accepted_programs'],1)
        self.assertEqual(result['metrics']['recurrent_2']['unseen_period']['verified_joint_traces'],0)

if __name__=='__main__':unittest.main()
