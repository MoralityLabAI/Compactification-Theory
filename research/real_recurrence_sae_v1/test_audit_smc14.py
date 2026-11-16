"""SMC-14 independent audit tests, standard library only."""
import copy
import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_smc14 import (
    cosine,calibration_sign,bootstrap,deterministic_shuffle_null,
    digit_rowband_null,audit,verify_puzzle,orientation_audit
)


class AuditSMC14Tests(unittest.TestCase):
    def test_puzzle_centering_and_orientation_replay(self):
        x=[1,2,3,4,5,6,7,8]
        y=[-z*2+13 for z in x]
        self.assertAlmostEqual(cosine(x,y),-1.)
        self.assertAlmostEqual(cosine([z+99 for z in x],y),-1.)
        self.assertAlmostEqual(cosine(x,[-z for z in y]),1.)
        self.assertIsNone(cosine([1]*8,y))
        orientation=calibration_sign([x,x],[y,y],[[True]*8,[True]*8])
        self.assertEqual(orientation['sign'],-1)
        self.assertFalse(orientation['unidentified'])
        with self.assertRaises(ValueError):
            cosine(x,y[:-1])

    def test_deterministic_within_puzzle_null(self):
        x=[1,2,3,4,5,6,7,8];y=x[:]
        null=deterministic_shuffle_null(x,y,[True]*8,1,0,11)
        self.assertEqual(null,deterministic_shuffle_null(x,y,[True]*8,1,0,11))
        self.assertLess(null,0.95)

    def test_cluster_bootstrap_fixed_seed(self):
        x=[0.1]*64
        out=bootstrap(x,seed=7,draws=200)
        self.assertAlmostEqual(out['mean'],0.1)
        self.assertEqual(out['n_independent_puzzles'],64)
        self.assertTrue(all(abs(z-0.1)<1e-12 for z in out['puzzle_bootstrap95']))
        self.assertIsNone(bootstrap([],seed=2))

    def test_posthoc_digit_band_null_preserves_puzzle_level_design(self):
        rows=[]
        for p in range(64):
            values=[float((i+p)%9+1) for i in range(8)]
            rows.append({
               'puzzle_index':p,
               'matched_effect_cosine':cosine(values,values),
               'source_causal_response':values,
               'target_matched_causal_response':values[:],
               'source_strict_top1_selection_mask':[True]*8,
               'selected_goal_positions':[i*9+(p%3) for i in range(8)],
               'selected_goal_truth_digits':[int((p+i)%9+1) for i in range(8)],
            })
        pair={'calibration':{'orientation':{'matched':{'sign':1}}},
              'heldout':{'test_novel_30':{'per_puzzle':rows}}}
        out=digit_rowband_null(pair,'test_novel_30',0,reps=6)
        self.assertEqual(out['status'],'posthoc_evaluation_only_null')
        self.assertEqual(out['tested_puzzles'],64)
        self.assertGreater(out['groups'],0)
        self.assertEqual(out['shuffle_reps_per_puzzle'],6)
        # Digit-only causal signals are preserved by this conditional null.
        self.assertAlmostEqual(out['matched_mean'],
                               out['digit_rowband_shuffled_mean'],places=12)

    def test_reject_invalid_source_or_partial_receipt(self):
        with self.assertRaises((KeyError,ValueError)):
            audit({'schema':'SMC-14/calibration-locked-cross-teacher-causal-transfer-v1',
                   'source_teacher_sha256':'forged'})
        with self.assertRaises(ValueError):
            verify_puzzle({
              'source_strict_top1_selection_mask':[True]*8,
              'selected_goal_positions':[0]*8,
              'selected_goal_truth_digits':[1]*8}, {},0)


if __name__=='__main__':
    unittest.main()
