from copy import deepcopy
from fractions import Fraction as Q
import unittest
from replay import check_case


def fixture():
    layers=[{'input_radius_upper':'1','jacobian_upper':'1','hessian_upper':'0','local_defect_upper':'0'} for _ in range(12)]
    clean=[{'rival':k,'margin_lower':'1','gradient_l1_full_upper':'0','gradient_l1_patch_upper':'0','adjoint_l2_upper':['1']*12} for k in range(1,10)]
    final=[{'rival':k,'remainder_upper':'0','margin_lower':'1'} for k in range(1,10)]
    return {'winner':0,'steps_after_anchor':3,'pairs':clean,'regions':[{'mode':'full','radius':'1/100','layers':layers,'pairs':final,'certified':True}]}


class TestReplay(unittest.TestCase):
    def test_valid_scalar_witness(self):self.assertTrue(check_case(fixture())[0][2])
    def test_reject_missing_rival(self):
        d=fixture();d['pairs'].pop()
        with self.assertRaises(ValueError):check_case(d)
    def test_reject_hidden_remainder(self):
        d=fixture();d['regions'][0]['layers'][0]['hessian_upper']='1';d['regions'][0]['layers'][0]['local_defect_upper']='1/2'
        with self.assertRaises(ValueError):check_case(d)
    def test_reject_negative_curvature(self):
        d=fixture();d['regions'][0]['layers'][0]['hessian_upper']='-1'
        with self.assertRaises(ValueError):check_case(d)
    def test_reject_shrunken_input_ball(self):
        d=fixture();d['regions'][0]['layers'][0]['input_radius_upper']='0'
        with self.assertRaises(ValueError):check_case(d)
    def test_reject_overstated_margin(self):
        d=fixture();d['regions'][0]['pairs'][0]['margin_lower']='2'
        with self.assertRaises(ValueError):check_case(d)
    def test_reject_false_flag(self):
        d=fixture();d['regions'][0]['certified']=False
        with self.assertRaises(ValueError):check_case(d)

if __name__=='__main__':unittest.main()
