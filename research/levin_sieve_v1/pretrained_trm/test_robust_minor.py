"""Focused unit tests for LSV-14 uniform-radius rank proof conditions."""
import copy
import unittest
from fractions import Fraction
import torch
from flint import arb, ctx
from robust_minor import (interval_state, check_single_receipt, check_full_receipt,
                          robust_minor)
from upstream_arch import TinyRecursiveModel


class RobustPositiveRadius(unittest.TestCase):
    def setUp(self):
        ctx.prec=160
        torch.set_num_threads(1)

    def test_full_ambient_input_box_exact_binary_midpoints(self):
        x=torch.zeros((81,48),dtype=torch.float32)
        x[3,5]=1.25
        st=interval_state(x,[0,1,2,3],'1/1000000')
        self.assertTrue(st.value[3,5].contains(arb('1.2500005')))
        self.assertTrue(st.value[7,7].contains(arb('-0.0000005')))
        self.assertFalse(st.value[3,5].contains(arb('1.25001')))
        self.assertEqual(len(st.tangents),4)
        self.assertTrue(st.tangents[0][0,0].contains(arb(1)))

    def test_reject_bad_radius_and_coordinates(self):
        x=torch.zeros((81,48),dtype=torch.float32)
        for rad in ('-0.01','2','garbage'):
            with self.assertRaises(ValueError):interval_state(x,[0,1,2,3],rad)
        for cols in ([0,0,2,3],[0,1,2,3888]):
            with self.assertRaises(ValueError):interval_state(x,cols,'1e-12')

    def test_independent_rational_robust_minor_nonzero(self):
        row={'radius':'1e-12','initial_box_dimensions':3888,'selected_columns':[0,1,2,3],
             'entry_intervals':[[f'{int(i==j)}' for j in range(4)] for i in range(4)]}
        proved,bound=check_single_receipt(row)
        self.assertTrue(proved)
        self.assertEqual(bound,Fraction(1))

    def test_independent_rational_rejects_ambiguous_minor(self):
        row={'radius':'1e-12','initial_box_dimensions':3888,'selected_columns':[0,1,2,3],
             'entry_intervals':[[f'{int(i==j)}' for j in range(4)] for i in range(4)]}
        row['entry_intervals'][0][0]='[1 +/- 2]'
        self.assertFalse(check_single_receipt(row)[0])

    def test_reject_crafted_invalid_full_receipt(self):
        row={'radius':'1e-12','initial_box_dimensions':3888,'selected_columns':[0,1,2,3],
             'entry_intervals':[[f'{int(i==j)}' for j in range(4)] for i in range(4)],
             'case_index':0}
        rec={'schema':'LSV-14/positive-radius-goal-rank-v1','radius':'1e-12','cases':[row]}
        self.assertEqual(check_full_receipt(rec)['positive_radius_rank_four_cases'],1)
        rec['radius']='-1'
        with self.assertRaises(ValueError):check_full_receipt(rec)
        rec['radius']='1e-12';rec['cases'][0]['case_index']=4
        with self.assertRaises(ValueError):check_full_receipt(rec)

    def test_nonlinear_model_small_ball_and_nonzero_at_zero(self):
        torch.manual_seed(4)
        model=TinyRecursiveModel().double().eval()
        z=torch.randn(81,48,dtype=torch.float64)*0.2
        # Finite positive radius on full box, regardless of determinant outcome
        # for an untrained random model. Must produce all 16 sound intervals.
        c=robust_minor(model,z,[0,1,2,3],0,1,2,'1e-16')
        self.assertEqual(len(c['entry_intervals']),4)
        self.assertEqual(c['initial_box_dimensions'],3888)
        self.assertEqual(c['radius'],'1e-16')


if __name__=='__main__':unittest.main()
