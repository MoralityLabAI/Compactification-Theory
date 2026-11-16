"""SMC-20 independent stdlib mutation tests for forged physical gradients and gates."""
import copy
import hashlib
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc20 as audit
from test_smc20_anchored_refit import SMC20AnchoredGradientTests


class SMC20AuditTests(unittest.TestCase):
    def toy_case(self):
        source_cols=[[0.]*48 for _ in range(96)]
        target_cols=[[0.]*48 for _ in range(96)]
        source_cols[5][0]=1.
        target_cols[7][0]=1.;target_cols[7][1]=1.
        active=[3,7,11,15]
        msg='SMC15|0|smc20_novel_30|202620030000|0|7'
        choices=[x for x in active if x!=7]
        other=choices[int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8],'big')%len(choices)]
        target_cols[other][1]=1.
        da=[0.]*48;da[0]=1.5
        db=[0.]*48;db[0]=1.;db[1]=1.
        dc=[0.]*48;dc[1]=math.sqrt(2.)
        g=[0.]*48;g[0]=1.;g[1]=2.
        src=1.;obs=2.2;other_obs=.8
        pg=src+audit.dot(g,audit.sub(db,da))
        po=src+audit.dot(g,audit.sub(dc,da))
        row={
          'seed':202620030000,
          'source_selected_ids':[5]+[0]*7,
          'target_matched_ids':[7]+[0]*7,
          'refit_other_active_ids':[other]+[0]*7,
          'refit_active_feature_ids':[active]+[[] for _ in range(7)],
          'reference_top1_coefficients':[1.5]+[0.]*7,
          'reference_top2_coefficients':[0.5]+[0.]*7,
          'refit_matched_coefficients':[1.]+[0.]*7,
          'refit_other_coefficients':[1.]+[0.]*7,
        }
        item={
          'reference_feature':5,'refit_matched_feature':7,
          'refit_other_feature':other,
          'reference_selected_coefficient':1.5,
          'refit_selected_coefficient':1.,
          'refit_other_selected_coefficient':1.,
          'reference_delta_48':da,
          'target_delta_48':db,'other_active_delta_48':dc,
          'clean_task_gradient_48':g,
          'physical_edit_distance':audit.norm(audit.sub(db,da)),
          'reference_physical_dose':audit.norm(da),
          'refit_physical_dose':audit.norm(db),
          'other_physical_dose':audit.norm(dc),
          'physical_edit_cosine':audit.dot(da,db)/(audit.norm(da)*audit.norm(db)),
          'observed_reference_drop':src,
          'observed_target_drop':obs,
          'observed_other_active_drop':other_obs,
          'gradient_anchored_prediction':pg,
          'reference_only_prediction':src,
          'other_active_surrogate_prediction':po,
          'direct_clean_linear_refit_prediction':audit.dot(g,db),
          'reference_only_absolute_error':abs(obs-src),
          'other_surrogate_absolute_error':abs(obs-po),
          'gradient_anchored_absolute_error':abs(obs-pg),
          'absolute_error_gain_over_reference':abs(obs-src)-abs(obs-pg),
          'absolute_error_gain_over_other':abs(obs-po)-abs(obs-pg),
          'reference_only_predicted_sign':1,
          'other_surrogate_predicted_sign':audit.sign_eps(po),
          'gradient_anchored_predicted_sign':audit.sign_eps(pg),
          'observed_target_sign':1,
          'clean_correct':True,
          'source_clean_correct_to_wrong':False,
          'refit_clean_correct_to_wrong':False,
          'other_clean_correct_to_wrong':False,
        }
        return item,row,source_cols,target_cols

    def test_independent_physical_gradient_and_edit_replay(self):
        item,row,source_cols,target_cols=self.toy_case()
        audit.audit_goal(item,row,0,'smc20_novel_30',0,source_cols,target_cols)

    def test_forged_gradient_predictor_is_detected(self):
        item,row,sc,tc=self.toy_case()
        wrong=copy.deepcopy(item)
        wrong['gradient_anchored_prediction']+=1.
        with self.assertRaisesRegex(ValueError,'forged'):
            audit.audit_goal(wrong,row,0,'smc20_novel_30',0,sc,tc)

    def test_forged_decoder_edit_is_detected(self):
        item,row,sc,tc=self.toy_case()
        wrong=copy.deepcopy(item)
        wrong['target_delta_48'][0]+=1.
        with self.assertRaisesRegex(ValueError,'native decoder'):
            audit.audit_goal(wrong,row,0,'smc20_novel_30',0,sc,tc)

    def test_changing_seeded_other_active_fails(self):
        item,row,sc,tc=self.toy_case()
        wrong=copy.deepcopy(item)
        wrong['refit_other_feature']=7
        with self.assertRaises(ValueError):
            audit.audit_goal(wrong,row,0,'smc20_novel_30',0,sc,tc)

    def test_replayed_bootstrap_detects_baseline_failure(self):
        tester=SMC20AnchoredGradientTests()
        rows=tester.toy_rows()
        prior=audit.independent_score(rows,202620130)
        self.assertTrue(prior['registered_success'])
        self.assertEqual(prior['puzzle_cluster_bootstrap95_gain_over_reference'],[1.,1.])
        self.assertEqual(prior['puzzle_cluster_bootstrap95_gain_over_other'],[2.,2.])
        for row in rows:
            if row['metrics'][0] is not None:
                row['metrics'][0]['gradient_anchored_prediction']=-2.
        self.assertFalse(audit.independent_score(rows,202620130)['registered_success'])

    def test_invalid_original_sha_and_registry_fail_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'frozen.json').write_text('{}')
            with self.assertRaises(ValueError):
                audit.audit({'schema':'not SMC20'},temp)
            with self.assertRaises(ValueError):
                audit.audit({'schema':'SMC-20/source-frozen-anchored-task-gradient-refit-v1',
                             'pre_registration_commit':'0'*40},temp)


if __name__=='__main__':unittest.main()
