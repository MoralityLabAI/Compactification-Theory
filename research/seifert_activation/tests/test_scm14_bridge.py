"""SCM-14 integration gates and conditional affine geometry."""
import copy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scm14_bridge import conditional_affine_bridge


def synthetic_hessian(e0='1/100',e1='1/100'):
    return {'schema':'scm14/two_coordinate_source_hessian/v1',
            'scope':'ideal_real_RMP_weights_conditional_CPU_binary64_BLAS_center_verifier',
            'hessian_upper_certified_under_source_contract':True,
            'intervention_coordinate':'u_in_unit_disk; patch=amplitude*axes@u',
            'observed_token_positions':[2,4],
            'per_visit':[{'visit':1,'observed':{
                '2':{'taylor_remainder_on_unit_disk_upper':e0},
                '4':{'taylor_remainder_on_unit_disk_upper':e1}}}],
            'trained_rmp_model_measured':False}


class BridgeTests(unittest.TestCase):
    def test_small_nonzero_winding_conditional(self):
        p=conditional_affine_bridge(synthetic_hessian(),visit=1,
                base=((0,0),(1,0)),
                jacobians=(((0,0),(0,0)),((2,0),(0,2))))
        self.assertEqual(p['result'],'conditional_affine_braid_preserved')
        self.assertEqual(p['per_pair']['0,1']['pair_winding'],1)
        self.assertFalse(p['model_level_topology_certified'])
        self.assertFalse(p['exact_affine_anchors_and_jacobians_verified'])

    def test_small_null_homotopy_conditionally(self):
        p=conditional_affine_bridge(synthetic_hessian(),visit=1,
                base=((0,0),(3,0)),
                jacobians=(((0,0),(0,0)),((1,0),(0,1))))
        self.assertEqual(p['result'],'conditional_full_null_braid')
        self.assertFalse(p['model_level_topology_certified'])

    def test_missing_source_claim_refused(self):
        r=synthetic_hessian();r['hessian_upper_certified_under_source_contract']=False
        with self.assertRaisesRegex(ValueError,'Missing source'):
            conditional_affine_bridge(r,visit=1,base=((0,0),(3,0)),
                           jacobians=(((0,0),(0,0)),((1,0),(0,1))))

    def test_source_coordinate_mismatch_refused(self):
        r=synthetic_hessian();r['intervention_coordinate']='unknown radius'
        with self.assertRaisesRegex(ValueError,'coordinate mismatch'):
            conditional_affine_bridge(r,visit=1,base=((0,0),(3,0)),
                           jacobians=(((0,0),(0,0)),((1,0),(0,1))))

    def test_negative_hessian_refused(self):
        with self.assertRaisesRegex(ValueError,'Negative'):
            conditional_affine_bridge(synthetic_hessian('-1/100'),visit=1,
                           base=((0,0),(3,0)),
                           jacobians=(((0,0),(0,0)),((1,0),(0,1))))

    def test_unverified_numeric_json_float_refused(self):
        with self.assertRaisesRegex(ValueError,'exact rational'):
            conditional_affine_bridge(synthetic_hessian(.05),visit=1,
                           base=((0,0),(3,0)),
                           jacobians=(((0,0),(0,0)),((1,0),(0,1))))

    def test_missing_visit_refused(self):
        with self.assertRaisesRegex(ValueError,'Visit'):
            conditional_affine_bridge(synthetic_hessian(),visit=2,
                           base=((0,0),(3,0)),
                           jacobians=(((0,0),(0,0)),((1,0),(0,1))))

    def test_mismatched_strands_refused(self):
        with self.assertRaisesRegex(ValueError,'match'):
            conditional_affine_bridge(synthetic_hessian(),visit=1,
                           base=((0,0),(3,0),(4,0)),
                           jacobians=(((0,0),(0,0)),((1,0),(0,1))))


if __name__=='__main__':unittest.main()
