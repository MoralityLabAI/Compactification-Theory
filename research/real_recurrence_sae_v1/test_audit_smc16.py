"""SMC-16 independent stdlib receipt auditor regression suite."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc16 as a


class IndependentAuditTests(unittest.TestCase):
    def sample(self):
        x=[0]*30+[1]*51
        y=[1]*81
        slots=[0,3,7,11,15,18,22,26]
        diag={'status':'eligible','killed_feature_reencoded':0.,
              'other_active_max_drift':0.,'sham_active_max_drift':0.,
              'original_active_count':4,'native_decoder_dose':1.,
              'kill_dose':1.5,'sham_dose':1.5,'native_dose_multiple':1.5}
        return {'puzzle_index':0,'seed':12,'goal_positions':slots,
                'goal_truth_digits':[1]*8,
                'source_top_feature_ids':[0]*8,
                'target_matched_feature_ids':[3]*8,
                'eligible_mask':[True]*8,
                'edit_diagnostics':[copy.deepcopy(diag) for _ in range(8)],
                'kill_margin_drop':[2.]*8,
                'sham_margin_drop':[0.]*8,
                'clean_correct':[True]*8,
                'kill_correct_to_wrong':[True]*8,
                'sham_correct_to_wrong':[False]*8}, {'seed':12,'x':x,'target':y}, [3]+[v for v in range(96) if v!=3]

    def test_eligibility_masks_dose_source_mapping_and_reencoding_guards(self):
        row,puzzle,assignment=self.sample()
        self.assertEqual(a.check_raw([row],[puzzle],assignment)[1],8)
        bad=copy.deepcopy(row)
        bad['edit_diagnostics'][0]['sham_dose']=.25
        with self.assertRaisesRegex(ValueError,'dose'):a.check_raw([bad],[puzzle],assignment)
        bad=copy.deepcopy(row)
        bad['target_matched_feature_ids'][0]=4
        with self.assertRaisesRegex(ValueError,'mapping'):a.check_raw([bad],[puzzle],assignment)
        bad=copy.deepcopy(row)
        bad['edit_diagnostics'][1]['killed_feature_reencoded']=1
        with self.assertRaisesRegex(ValueError,'encoder'):a.check_raw([bad],[puzzle],assignment)
        bad=copy.deepcopy(row)
        bad['eligible_mask'][0]=False
        with self.assertRaisesRegex(ValueError,'eligibility'):a.check_raw([bad],[puzzle],assignment)

    def test_independent_puzzle_bootstrap_and_flip_gate(self):
        row,_,_=self.sample()
        test=[]
        for i in range(64):
            r=copy.deepcopy(row)
            r['eligible_mask']=[i<24]+[False]*7
            test.append(r)
        stats=a.independent_score(test,seed=3)
        self.assertTrue(stats['registered_success'])
        self.assertEqual(stats['eligible_goals'],24)
        self.assertEqual(stats['nonempty_puzzle_clusters'],24)
        self.assertEqual(stats['mean_selective_kill_minus_sham_margin_drop'],2.)
        for r in test:r['kill_correct_to_wrong'][0]=False
        self.assertFalse(a.independent_score(test,seed=3)['registered_success'])

    def test_avoid_imputing_zero_on_ineligible_goals(self):
        row,puzzle,assignment=self.sample()
        row['eligible_mask'][0]=False
        row['edit_diagnostics'][0]={'status':'matched_inactive'}
        row['kill_margin_drop'][0]=None
        row['sham_margin_drop'][0]=None
        row['kill_correct_to_wrong'][0]=None
        row['sham_correct_to_wrong'][0]=None
        self.assertEqual(a.check_raw([row],[puzzle],assignment)[1],7)
        row['kill_margin_drop'][0]=0.
        with self.assertRaisesRegex(ValueError,'abstain'):
            a.check_raw([row],[puzzle],assignment)

    def test_all_seven_source_mappings_are_independently_pinned(self):
        valid={'all_seven_original_smc13c_hungarian_assignments_sha_verified':True,
               'all_seven_original_smc13c_mapping_sha256':[
                 {'matched_assignment_sha256':a.ORIGINAL_CALIB_SHA['matched'][i],
                  'clue_shuffled_assignment_sha256':a.ORIGINAL_CALIB_SHA['clue_shuffled'][i],
                  'matched_to_original_smc13c_receipt':True} for i in range(7)]}
        a.check_mappings(valid)
        altered=copy.deepcopy(valid)
        altered['all_seven_original_smc13c_mapping_sha256'][4]['matched_assignment_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'fingerprint'):a.check_mappings(altered)


if __name__=='__main__':unittest.main()
