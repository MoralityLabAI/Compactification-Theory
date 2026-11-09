"""Independent finite-count, information bound, recurrent policy and anti-leak tests."""
from __future__ import annotations

from pathlib import Path
import json
import math
import unittest

try:
    from .active03 import (PREFIX_DEGREE, HERE, baseline, candidate_defect,
                          mutant_coefficient, query_bit, posterior_interval,
                          midpoint_policy, static_policy, fixed_thresholds,
                          run_policy, all_policy_metrics, optimum_success_mass,
                          adaptive_success_bound, static_success_bound, validate_series)
    from .student03 import QueryStudent, packed_inputs, rnn_inputs, PARAMETERS
    from .train_active03 import generate_histories, TRAIN_SIZES, HELDOUT_SIZES
    from .bench_active03 import blinded, run_replay
except ImportError:
    from active03 import (PREFIX_DEGREE, HERE, baseline, candidate_defect,
                          mutant_coefficient, query_bit, posterior_interval,
                          midpoint_policy, static_policy, fixed_thresholds,
                          run_policy, all_policy_metrics, optimum_success_mass,
                          adaptive_success_bound, static_success_bound, validate_series)
    from student03 import QueryStudent, packed_inputs, rnn_inputs, PARAMETERS
    from train_active03 import generate_histories, TRAIN_SIZES, HELDOUT_SIZES
    from bench_active03 import blinded, run_replay


class ExactQueryTests(unittest.TestCase):
    def test_same_observed_prefix_and_independent_counting_oracles(self):
        for size in (7,13,33):
            r=validate_series(size)
            self.assertTrue(r['independent_oracles_match'])
            self.assertEqual(r['prefix_identical_through'],64)
            self.assertEqual(r['binary_comparisons_checked'],size*(size-1))

    def test_every_query_is_binary_threshold(self):
        for size in (2,5,13,33,65):
            for idx in range(size):
                self.assertEqual(candidate_defect(idx,size),
                                 None if idx==size-1 else PREFIX_DEGREE+idx+1)
                for t in range(size-1):
                    self.assertEqual(query_bit(idx,size,t),int(idx<size-1 and idx<=t))

    def test_mutant_first_disagreement(self):
        n=96
        f=baseline(n)
        for idx in (0,8,15,31):
            k=candidate_defect(idx,33)
            for degree in (65,k-1,k,n):
                if degree<65:continue
                got=mutant_coefficient(idx,33,degree)
                self.assertEqual(got,f[degree]+(f[degree-k] if degree>=k else 0))
                self.assertEqual(got>f[degree],degree>=k)
        self.assertEqual(mutant_coefficient(32,33,n),f[n])

    def test_posterior_query_updates_and_mutation(self):
        self.assertEqual(posterior_interval((),33),(0,32))
        self.assertEqual(posterior_interval(((15,0),),33),(16,32))
        self.assertEqual(posterior_interval(((15,0),(23,1)),33),(16,23))
        self.assertEqual(posterior_interval(((15,0),(23,0)),33),(24,32))
        with self.assertRaises(ValueError):
            posterior_interval(((0,1),(0,0)),33)
        with self.assertRaises(ValueError):
            posterior_interval(((45,1),),33)

    def test_active_teacher_matches_uniform_bellman_and_static_bound(self):
        for size in (5,9,13,33,49):
            for budget in range(7):
                adapt=all_policy_metrics(size,budget,midpoint_policy)
                static=all_policy_metrics(size,budget,static_policy)
                self.assertEqual(adapt['correct'],adaptive_success_bound(size,budget))
                self.assertEqual(static['correct'],static_success_bound(size,budget))
                self.assertEqual(adapt['correct'],optimum_success_mass((1,)*size,budget))
                self.assertLessEqual(static['transcripts'],budget+1)
                self.assertLessEqual(adapt['transcripts'],2**budget)

    def test_sharp_33_case(self):
        self.assertEqual([all_policy_metrics(33,b,midpoint_policy)['correct'] for b in range(7)],
                         [1,2,4,8,16,32,33])
        self.assertEqual([all_policy_metrics(33,b,static_policy)['correct'] for b in range(7)],
                         [1,2,3,4,5,6,7])
        self.assertEqual(fixed_thresholds(33,4),(5,12,18,25))
        self.assertEqual(run_policy(32,33,6,midpoint_policy).posterior,(32,32))

    def test_skewed_prior_bellman_oracle(self):
        self.assertEqual(optimum_success_mass((5,1,1),0),5)
        self.assertEqual(optimum_success_mass((5,1,1),1),6)
        self.assertEqual(optimum_success_mass((5,1,1),2),7)
        with self.assertRaises(ValueError):
            optimum_success_mass((0,1),1)

    def test_invalid_queries_are_rejected(self):
        with self.assertRaises(ValueError):
            query_bit(0,33,-1)
        with self.assertRaises(ValueError):
            candidate_defect(33,33)
        with self.assertRaises(ValueError):
            run_policy(0,33,7,midpoint_policy)
        with self.assertRaises(ValueError):
            run_policy(0,33,1,lambda h,s,b:32)


class PortableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rnn=QueryStudent.load(HERE/'results/rrc03_recurrent.json')
        cls.ff=QueryStudent.load(HERE/'results/rrc03_feedforward.json')

    def test_exact_parameter_parity(self):
        self.assertEqual(self.rnn.checkpoint['parameter_count'],PARAMETERS)
        self.assertEqual(self.ff.checkpoint['parameter_count'],PARAMETERS)
        self.assertEqual(PARAMETERS,353)

    def test_model_input_contains_only_public_history(self):
        history=((15,1),(7,0),(11,1))
        self.assertEqual(len(rnn_inputs(history,33)),3)
        self.assertEqual(len(packed_inputs(history,33)),20)
        for model in (self.rnn,self.ff):
            p=model.predict_fraction(history,33)
            self.assertTrue(0<p<1)
            self.assertIn(model.action(history,33,4),range(32))
        with self.assertRaises(ValueError):
            packed_inputs(((4,3),),33)

    def test_disjoint_train_and_heldout_hypothesis_sizes(self):
        self.assertFalse(set(TRAIN_SIZES)&set(HELDOUT_SIZES))
        a=generate_histories(seed=11,per_size=8)
        b=generate_histories(seed=11,per_size=8)
        self.assertEqual(a,b)
        self.assertTrue(all(size in TRAIN_SIZES for _,size,_ in a))

    def test_recurrent_policy_uses_feedback(self):
        with_feedback=all_policy_metrics(33,4,self.rnn.action)
        without_feedback=all_policy_metrics(33,4,blinded(self.rnn.action))
        self.assertEqual(with_feedback['correct'],16)
        self.assertEqual(without_feedback['correct'],5)
        self.assertEqual(all_policy_metrics(49,4,self.rnn.action)['correct'],16)
        self.assertEqual(all_policy_metrics(49,4,self.ff.action)['correct'],16)

    def test_recurrent_not_superior_in_every_long_budget(self):
        r=all_policy_metrics(49,6,self.rnn.action)
        f=all_policy_metrics(49,6,self.ff.action)
        self.assertEqual((r['correct'],f['correct']),(40,42))

    def test_checkpoint_schema_and_shape_rejection(self):
        cp=dict(self.rnn.checkpoint)
        cp['parameter_count']=999
        with self.assertRaises(ValueError):
            QueryStudent(cp)
        cp=dict(self.ff.checkpoint)
        cp['schema']='wrong'
        with self.assertRaises(ValueError):
            QueryStudent(cp)

    def test_receipt_source_and_full_deterministic_replay(self):
        stored=json.loads((HERE/'results/rrc03_pilot.json').read_text())
        rebuilt=run_replay()
        self.assertEqual(stored,rebuilt)
        self.assertEqual(len(rebuilt['policies']),28)
        self.assertEqual(set(rebuilt['heldout_sizes']),{13,25,49})


if __name__ == '__main__':unittest.main()
