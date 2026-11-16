"""Independent stdlib receipt audit contract tests using synthetic frozen data."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc15 as a


class AuditSMC15Tests(unittest.TestCase):
    def test_reorients_original_calibration_only(self):
        rows=[{'eligible_mask':[True,True]+[False]*6,
               'source_drop':[1.,-2.]+[0.]*6,
               'matched_drop':[-3.,4.]+[0.]*6,
               'other_active_drop':[5.,-6.]+[0.]*6}]
        signs=a.orient_replay(rows)
        self.assertEqual(signs['matched']['sign'],-1)
        self.assertEqual(signs['other']['sign'],1)
        self.assertEqual(signs['matched']['eligible_calibration_goals'],2)

    def test_puzzle_cluster_bootstrap_rejects_missing_coverage(self):
        rows=[]
        for p in range(64):
            rows.append({'eligible_mask':[p<20,p<20]+[False]*6,
                         'source_drop':[1.]*8,'matched_drop':[2.]*8,
                         'other_active_drop':[-3.]*8})
        signs={'matched':{'sign':1,'identified':True},'other':{'sign':1,'identified':True}}
        result=a.score_replay(rows,signs,3)
        self.assertTrue(result['registered_success'])
        self.assertEqual(result['eligible_goals'],40)
        self.assertEqual(result['nonempty_puzzle_clusters'],20)
        self.assertAlmostEqual(result['matched_minus_other_active'],2.)
        absent=a.score_replay(rows,{**signs,'matched':{'sign':1,'identified':False}},3)
        self.assertFalse(absent['registered_success'])

    def test_fail_on_dose_or_inactive_control(self):
        puzzle={'seed':7,'x':[0]*30+[1]*51,'target':[1]*81}
        row={'puzzle_index':0,'seed':7,'goal_positions':[0,3,7,11,15,18,22,26],
             'eligible_mask':[True]*8,'source_selected_ids':[0]*8,
             'target_matched_ids':[3]*8,'target_other_active_ids':[4]*8,
             'target_other_active_coefficients':[2.]*8,
             'target_match_clean_coefficients':[3.]*8,
             'target_match_after_erase_coefficients':[1.]*8,
             'matched_physical_dose':[3.]*8,'other_physical_dose':[3.]*8,
             'source_drop':[1.]*8,'matched_drop':[1.]*8,'other_active_drop':[1.]*8,
             'source_clean_correct':[True]*8,'target_clean_correct':[True]*8,
             'source_correct_to_wrong_erase':[False]*8,
             'target_matched_correct_to_wrong_erase':[False]*8,
             'target_other_correct_to_wrong_erase':[False]*8}
        a.validate_raw([row],[puzzle],True)
        tamper=copy.deepcopy(row);tamper['other_physical_dose'][3]=.5
        with self.assertRaisesRegex(ValueError,'physical dose'):
            a.validate_raw([tamper],[puzzle],True)
        tamper=copy.deepcopy(row);tamper['target_other_active_coefficients'][0]=0
        with self.assertRaisesRegex(ValueError,'naturally active'):
            a.validate_raw([tamper],[puzzle],True)

    def test_negative_result_does_not_promote_to_success(self):
        rows=[{'eligible_mask':[True]*8,'source_drop':[1.]*8,
               'matched_drop':[-1.]*8,'other_active_drop':[1.]*8} for _ in range(64)]
        signs={'matched':{'sign':1,'identified':True},'other':{'sign':1,'identified':True}}
        res=a.score_replay(rows,signs,19)
        self.assertFalse(res['registered_success'])
        self.assertEqual(res['matched_minus_other_active'],-2.)


if __name__=='__main__':unittest.main()
