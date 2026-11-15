"""SMC-11C independent receipt audit rejects altered source and proof labels."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_smc11_joint import (
    CHECKPOINT,ROSTER,SCHEMA,BUDGETS,
    check_joint,guard,verify_selection
)


def synthetic_receipt():
    identity={
      'seed':71,'puzzle_id':3,'goal_token':0,'status':'certified',
      'top1_feature':7,'top4_features':[1,2,3,7],
      'top1_radius_sq_exact':'1/4','top4_radius_sq_exact':'1/9',
      'torch_same_top1_and_top4':True,
      'budget_guards':{
          eta:{'top1':guard('1/4',eta),'top4':guard('1/9',eta)}
          for eta in BUDGETS}
    }
    context={
        'seed':71,'puzzle_id':3,'goal_token':0,
        'radius':'1/5','common_l2_uncertainty':'0',
        'sae_snapshot_sha256':'a'*64,
        'exact_top1_guard':True,'exact_top4_guard':True,
        'comparisons':{
            'isotropic':'sae_active_greater',
            'sae_inactive':'control_greater',
            'jacobian_optimal':'control_greater',
        },
        'composed_policy_comparisons':{
            'isotropic':'sae_active_greater',
            'sae_inactive':'control_greater',
            'jacobian_optimal':'control_greater',
        },
    }
    summary={
       f'1/5|0|{control}':{
            'sae_active_greater':int(context['comparisons'][control]=='sae_active_greater'),
            'control_greater':int(context['comparisons'][control]=='control_greater'),
            'inconclusive_original':0,'abstain_selection':0,'total':1}
       for control in context['comparisons']}
    return {
        'schema':SCHEMA,'checkpoint_sha256':CHECKPOINT,'roster_sha256':ROSTER,
        'sae_steps':300,'arb_precision_bits':160,'test_puzzle_ids':[3],
        'seeds':[71],'radii':['1/5'],'common_l2_budget_grid':['0'],
        'frozen_sae_snapshots':[{
             'seed':71,'trained_sae_sha256':'a'*64,
             'snapshot_file_sha256':'b'*64
        }],
        'exact_goal_selection_cases':[identity],
        'cases':[context],
        'same_fit_composed_summary':summary,
    }


class JointReceiptAuditTests(unittest.TestCase):
    def test_valid_consistent_minimal_source_sample(self):
        receipt=synthetic_receipt()
        result=check_joint(receipt,full=False)
        self.assertTrue(result['all_source_fingerprint_provenance_matched'])
        self.assertEqual(result['rechecked_exact_selection_cells'],1)
        self.assertEqual(result['rechecked_arb_causal_contexts'],1)
        self.assertEqual(result['rechecked_control_decisions'],3)
        self.assertEqual(result['selection_abstentions'],0)

    def test_exhaustive_exact_rational_guard_with_boundary(self):
        self.assertTrue(guard('1/9','1/4'))
        self.assertFalse(guard('1/9','1/3'))
        self.assertTrue(guard(None,'100000000'))
        with self.assertRaises(ValueError):guard('1/9','-1/10')
        example=synthetic_receipt()['exact_goal_selection_cases'][0]
        example['budget_guards']['0']['top1']=False
        with self.assertRaises(ValueError):verify_selection(example)

    def test_reject_different_sae_weights_and_wrong_policy(self):
        x=synthetic_receipt()
        x['cases'][0]['sae_snapshot_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'different SAE'):
            check_joint(x,full=False)
        x=synthetic_receipt()
        x['cases'][0]['composed_policy_comparisons']['isotropic']='control_greater'
        with self.assertRaisesRegex(ValueError,'composition disagrees'):
            check_joint(x,full=False)

    def test_reject_missing_rows_and_incorrect_counts(self):
        x=synthetic_receipt()
        x['cases'].append(copy.deepcopy(x['cases'][0]))
        with self.assertRaisesRegex(ValueError,'duplicated causal tuple'):
            check_joint(x,full=False)
        x=synthetic_receipt()
        x['same_fit_composed_summary']['1/5|0|isotropic']['sae_active_greater']=0
        with self.assertRaisesRegex(ValueError,'summary mismatches'):
            check_joint(x,full=False)
        x=synthetic_receipt()
        x['checkpoint_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'untrusted source'):
            check_joint(x,full=False)

    def test_guard_composition_boundary_abstains_exactly(self):
        x=synthetic_receipt()
        x['cases'][0]['common_l2_uncertainty']='1/1000000'
        x['cases'][0]['exact_top1_guard']=True
        x['cases'][0]['exact_top4_guard']=True
        with self.assertRaisesRegex(ValueError,'unregistered causal radius or uncertainty'):
            check_joint(x,full=False)


if __name__=='__main__':
    unittest.main()
