"""RRC-07 finite first-fault algebra and source-blind neural controls."""
from __future__ import annotations
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from .rrc07 import (PREFIX,HORIZON,Case07,new_roster,raw_features,first_fault,fault_feedback,
                        score_proposal,roster_hash)
    from .rrc import (gap_coefficients,euler_exponents)
    from .rrc05 import (Program05,product_coefficients,exact_program)
    from .raw_student06 import infer_raw
    from .student07 import (parameter_count,portable_forward,propose_int)
except ImportError:
    from rrc07 import (PREFIX,HORIZON,Case07,new_roster,raw_features,first_fault,fault_feedback,
                      score_proposal,roster_hash)
    from rrc import (gap_coefficients,euler_exponents)
    from rrc05 import (Program05,product_coefficients,exact_program)
    from raw_student06 import infer_raw
    from student07 import (parameter_count,portable_forward,propose_int)

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY:
    try:
        from .bench07 import load
        from .fast07 import fast_base,fast_forward
    except ImportError:
        from bench07 import load
        from fast07 import fast_base,fast_forward
else:
    load = fast_base = fast_forward = None

HERE=Path(__file__).resolve().parent

class ExactFeedback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program=Program05(0,5,(1,4))
        cls.f=product_coefficients(HORIZON,cls.program)
        cls.a=euler_exponents(cls.f[:PREFIX+1])

    def test_first_fault_localization_and_exact_update(self):
        for n in (1,7,31,64):
            broken=self.a.copy()
            broken[n]+=(-2 if n%2 else 3)
            self.assertEqual(first_fault(self.f[:PREFIX+1],broken),n)
            fixed=fault_feedback(self.f[:PREFIX+1],broken,1)
            self.assertEqual(fixed['edits'],[{'degree':n,'delta':self.a[n]-broken[n]}])
            self.assertEqual(fixed['proposal'],self.a)
            self.assertEqual(fixed['first_fault'],None)
            self.assertEqual(fixed['program'],self.program)
            self.assertEqual(fixed['queried_residual_vectors'],2)

    def test_mixed_faults_monotonic_progress(self):
        broken=self.a.copy()
        for n in (2,5,20,63):broken[n]+=1
        last=0
        for budget in (0,1,2,3,4,8):
            result=fault_feedback(self.f[:PREFIX+1],broken,budget)
            self.assertEqual(result['corrected'],min(budget,4))
            expected=(2,5,20,63,None)[min(budget,4)]
            self.assertEqual(result['first_fault'],expected)
            if expected is not None:
                self.assertGreater(expected,last)
                last=expected
        self.assertEqual(fault_feedback(self.f[:PREFIX+1],broken,64)['proposal'],self.a)

    def test_two_different_oracles_and_rr_positive_controls(self):
        for minimum,res in ((1,(1,4)),(2,(2,3))):
            f=gap_coefficients(PREFIX,minimum_part=minimum)
            broken=[0]*(PREFIX+1)
            cleaned=fault_feedback(f,broken,64)
            self.assertTrue(cleaned['verified_prefix'])
            self.assertEqual(cleaned['program'],Program05(0,5,res))
            self.assertEqual(cleaned['proposal'],euler_exponents(f))

    def test_negative_control_rejects_incomplete_program(self):
        proposal=self.a.copy();proposal[9]+=1
        self.assertIsNone(fault_feedback(self.f[:PREFIX+1],proposal,0)['program'])
        self.assertEqual(fault_feedback(self.f[:PREFIX+1],proposal,1)['program'],self.program)

    def test_rejects_invalid_budgets_and_shapes(self):
        with self.assertRaises(ValueError):
            fault_feedback(self.f[:PREFIX+1],self.a,-1)
        with self.assertRaises(ValueError):
            fault_feedback(self.f[:PREFIX+1],self.a,65)
        with self.assertRaises(ValueError):
            first_fault(self.f[:PREFIX+1],self.a[:-1])
        with self.assertRaises(ValueError):
            fault_feedback(self.f[:PREFIX+1],[1]+self.a[1:],1)

    def test_collision_future_is_unidentifiable(self):
        cases=new_roster(train_size=12,dev_size=8,unseen_per_period=2)['collision']
        self.assertEqual(len(cases),10)
        for a,b in zip(cases[::2],cases[1::2]):
            self.assertEqual(a.prefix,b.prefix)
            self.assertNotEqual(a.full,b.full)
            self.assertEqual(fault_feedback(a.prefix,[0]*65,64)['proposal'],
                             fault_feedback(b.prefix,[0]*65,64)['proposal'])
            program=exact_program(a.prefix)
            if program is not None:
                predictions=tuple(product_coefficients(HORIZON,program))
                self.assertLessEqual(int(predictions==a.full)+int(predictions==b.full),1)

    def test_small_integer_series_complete_repair(self):
        for minimum in (1,2,3,4):
            f=gap_coefficients(PREFIX,gap=3,minimum_part=minimum)
            a=euler_exponents(f)
            raw=[0]*(PREFIX+1)
            repaired=fault_feedback(f,raw,64)
            self.assertEqual(repaired['proposal'],a)
            self.assertTrue(repaired['verified_prefix'])
            self.assertLessEqual(repaired['corrected'],64)

@unittest.skipUnless(HAS_NUMPY, "NumPy optional; RRC-07 CI installs it for neural parity")
class NeuralContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.models=load()
        cls.cases=new_roster(train_size=14,dev_size=6,unseen_per_period=2)

    def test_parameter_matching_and_frozen_source(self):
        self.assertEqual(parameter_count('recurrent'),4227)
        self.assertEqual(parameter_count('feedforward'),4297)
        self.assertLess(abs(4297-4227)/4227,.02)
        for ckpt in self.models.values():
            self.assertFalse(ckpt['exact_residual_at_inference'])
            self.assertFalse(ckpt['test_generated_in_training'])

    def test_fresh_train_dev_prefixes_disjoint(self):
        self.assertTrue({c.prefix for c in self.cases['train']}.isdisjoint(
                        {c.prefix for c in self.cases['dev']}))
        for key in ('unseen_moduli','unseen_cap'):
            self.assertTrue({c.prefix for c in self.cases['train']}.isdisjoint(
                            {c.prefix for c in self.cases[key]}))

    def test_new_roster_training_does_not_generate_test(self):
        subset=new_roster(train_size=8,dev_size=6,include_test=False)
        self.assertEqual(len(subset['train']),8)
        self.assertEqual(len(subset['dev']),6)
        self.assertEqual(sum(len(v) for k,v in subset.items() if k not in ('train','dev')),0)

    def test_model_has_no_future_dependence(self):
        a,b=self.cases['collision'][:2]
        for kind in self.models:
            logits=infer_raw(a.prefix,self.base)
            from_a=portable_forward(raw_features(a.prefix),logits,self.models[kind])
            from_b=portable_forward(raw_features(b.prefix),logits,self.models[kind])
            self.assertEqual(from_a,from_b)

    def test_eligible_passes_and_output_shape(self):
        c=self.cases['dev'][0]
        features=raw_features(c.prefix)
        logits=infer_raw(c.prefix,self.base)
        baseline=[row[:3] for row in logits]
        self.assertEqual(portable_forward(features,logits,self.models['recurrent'],passes=0),baseline)
        for depth in (1,2,4,8):
            out=portable_forward(features,logits,self.models['recurrent'],passes=depth)
            self.assertEqual(len(out),65)
            self.assertEqual(len(propose_int(out)),65)
        with self.assertRaises(ValueError):
            portable_forward(features,logits,self.models['recurrent'],passes=-1)
        with self.assertRaises(ValueError):
            portable_forward(features,logits,self.models['feedforward'],passes=2)

    def test_numpy_scalar_frozen_and_student_parity(self):
        for c in self.cases['dev'][:3]:
            features,base_fast=fast_base(c.prefix,self.base)
            base_scalar=infer_raw(c.prefix,self.base)
            self.assertLess(max(abs(a-b) for x,y in zip(base_fast,base_scalar)
                                for a,b in zip(x,y)),1e-10)
            for kind in self.models:
                depth_list=(0,1,2,4,8) if kind=='recurrent' else (None,)
                for depth in depth_list:
                    scalar=portable_forward(features,base_scalar,self.models[kind],passes=depth)
                    numpy=fast_forward(features,base_fast,self.models[kind],passes=depth)
                    self.assertLess(max(abs(a-b) for x,y in zip(scalar,numpy)
                                        for a,b in zip(x,y)),1e-10)
                    self.assertEqual(propose_int(scalar),propose_int(numpy))

    def test_exact_feedback_cached_budget_equivalence(self):
        c=self.cases['unseen_moduli'][0]
        proposal=propose_int([x[:3] for x in infer_raw(c.prefix,self.base)])
        whole=fault_feedback(c.prefix,proposal,64)
        edits=whole['edits'];total=len(edits)
        for budget in (0,1,2,4,8,16,32,64):
            partial=fault_feedback(c.prefix,proposal,budget)
            steps=min(budget,total)
            self.assertEqual(partial['edits'],edits[:steps])
            self.assertEqual(partial['queried_residual_vectors'],steps+1)
            if steps<total:
                self.assertEqual(partial['first_fault'],edits[steps]['degree'])
            else:
                self.assertIsNone(partial['first_fault'])

    def test_neural_output_is_a_proposal_needing_exact_verification(self):
        c=self.cases['unseen_moduli'][0]
        features=raw_features(c.prefix)
        base=infer_raw(c.prefix,self.base)
        for kind in self.models:
            z=portable_forward(features,base,self.models[kind])
            proposal=propose_int(z)
            metric=score_proposal(c,proposal)
            self.assertEqual(metric['accepted_program'],0 if metric['first_residual'] else
                             metric['accepted_program'])
            if metric['accepted_program']:
                self.assertEqual(metric['first_residual'],None)

if __name__=='__main__':unittest.main()
