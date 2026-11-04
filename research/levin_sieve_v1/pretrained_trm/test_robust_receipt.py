"""Tiny independent-replay tests, no Arb or Torch dependencies."""
import copy
import unittest
from fractions import Fraction
from verify_robust_receipt import verify
from robust_inverse import inverse_perturbation_witness


def fake():
    base={'schema':'LSV-13/arb-validated-goal-minors-v1',
          'checkpoint_sha256':'sha','roster_sha256':'roster',
          'rigorously_verified_rank_four_cases':64,
          'cases':[{'case_index':i,'observed_autograd_binary64_hex':
                    [['0x1.0p+0' if k==j else '0x0.0p+0' for j in range(4)] for k in range(4)],
                    'columns':[0,1,2,3],
                    'selected_goal_cell':0,'winner_digit':1,'rival_digit':2} for i in range(64)]}
    row={'case_index':0,'radius':'1e-20','initial_box_dimensions':3888,
         'selected_columns':[0,1,2,3],'goal_cell':0,'winner_digit':1,'rival_digit':2,
         'entry_intervals':[['1','0','0','0'],['0','1','0','0'],
                            ['0','0','1','0'],['0','0','0','1']],
         'independent_rational_det_excludes_zero':True,
         'independent_rational_abs_det_lower_bound':'1',
         'neumann_certificate':inverse_perturbation_witness(
             [['1','0','0','0'],['0','1','0','0'],['0','0','1','0'],['0','0','0','1']],
             base['cases'][0]['observed_autograd_binary64_hex']),
         'top1_stable_across_box':True,'top2_stable_across_box':True}
    rec={'schema':'LSV-14/positive-radius-goal-rank-v1',
         'checkpoint_sha256':'sha','roster_sha256':'roster',
         'radius':'1e-20','cases_checked':1,'cases':[row],
         'neumann_certified_cases':1,'rank_four_cases_by_either_method':1,
         'top1_stable_cases':1,'top2_stable_cases':1,
         'replay':{'positive_radius_rank_four_cases':1}}
    return rec,base

class IndependentProofReplay(unittest.TestCase):
    def test_proof_and_lock(self):
        record,base=fake()
        result=verify(record,base)
        self.assertEqual(result['positive_radius_rank_four_cases'],1)
        self.assertTrue(result['locked_selection_matches'])
        self.assertEqual(result['neumann_rank_four_cases'],1)

    def test_reject_falsified_det_interval(self):
        record,base=fake()
        record['cases'][0]['entry_intervals'][0][0]='[1 +/- 2]'
        with self.assertRaises(AssertionError):verify(record,base)

    def test_reject_changed_column(self):
        record,base=fake()
        record['cases'][0]['selected_columns']=[0,1,2,4]
        with self.assertRaises(AssertionError):verify(record,base)

    def test_reject_nonpositive_radius(self):
        record,base=fake()
        record['radius']='0'
        with self.assertRaises(ValueError):verify(record,base)

    def test_reject_changed_hash(self):
        record,base=fake()
        record['roster_sha256']='bad'
        with self.assertRaises(ValueError):verify(record,base)

if __name__=='__main__':unittest.main()
