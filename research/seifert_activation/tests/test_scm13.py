"""SCM-13 affine-disk topology and frozen 64-task campaign contracts."""
from __future__ import annotations
import copy
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scm13_affine_disk import affine_disk_certificate, taylor_errors_from_hessian
from scm13_receipt_integrity import strict_compare_files
from scm13_diagnostic import screen_frames, audit_capture
from knot import synthetic_braid
from scm12_compare import MODELS,CHARTS
from scm12_reference import source_record
from winding import pairwise_windings


def certificate(radius,base=None,matrices=None,errors=None,**kw):
    base=base if base is not None else [[0,0],[2,0]]
    matrices=matrices if matrices is not None else [[[0,0],[0,0]],[[1,0],[0,1]]]
    errors=errors if errors is not None else [0,0]
    return affine_disk_certificate(base,matrices,radius,errors,**kw)


def points(base,matrices,radius,frames=256,nonlinear=0.0):
    result=[]
    for k in range(frames+1):
        x=math.cos(2*math.pi*k/frames) if k<frames else 1.0
        y=math.sin(2*math.pi*k/frames) if k<frames else 0.0
        out=[]
        for i,(a,B) in enumerate(zip(base,matrices)):
            noise=(nonlinear*math.sin(3*2*math.pi*k/frames)) if k<frames else 0.
            out.append((a[0]+radius*(B[0][0]*x+B[0][1]*y)+noise,
                        a[1]+radius*(B[1][0]*x+B[1][1]*y)))
        result.append(out)
    return result


class AffineDiskTests(unittest.TestCase):
    def test_below_threshold_entire_braid_null(self):
        r=certificate(1)
        self.assertEqual(r['result'],'conditional_full_null_braid')
        self.assertTrue(r['full_affine_disk_homotopy_certified'])
        self.assertEqual(r['per_pair']['0,1']['pair_winding'],0)

    def test_above_threshold_nonzero_winding(self):
        r=certificate(3)
        self.assertEqual(r['result'],'conditional_affine_braid_preserved')
        self.assertEqual(r['per_pair']['0,1']['pair_winding'],1)

    def test_reflecting_response_reverses_winding(self):
        B=[[[0,0],[0,0]],[[1,0],[0,-1]]]
        self.assertEqual(certificate(3,matrices=B)['per_pair']['0,1']['pair_winding'],-1)

    def test_exact_boundary_collision_refused(self):
        r=certificate(2)
        self.assertEqual(r['result'],'inconclusive')
        self.assertIsNone(r['per_pair']['0,1']['pair_winding'])

    def test_error_less_than_clearance_accepted(self):
        self.assertEqual(certificate(1,errors=['1/4','1/4'])['result'],'conditional_full_null_braid')

    def test_error_at_clearance_refused(self):
        self.assertEqual(certificate(1,errors=['1/2','1/2'])['result'],'inconclusive')

    def test_rational_irrational_threshold(self):
        B=[[[0,0],[0,0]],[[1,0],[0,1]]]
        self.assertEqual(certificate('7/5',base=[[0,0],[1,1]],matrices=B)['per_pair']['0,1']['pair_winding'],0)
        self.assertEqual(certificate('3/2',base=[[0,0],[1,1]],matrices=B)['per_pair']['0,1']['pair_winding'],1)

    def test_rank_deficient_pair_inconclusive(self):
        B=[[[0,0],[0,0]],[[1,0],[0,0]]]
        r=certificate(1,matrices=B)
        self.assertEqual(r['result'],'inconclusive')
        self.assertEqual(r['per_pair']['0,1']['status'],'rank_deficient_unresolved')

    def test_null_three_strands_implies_nonabelian_null(self):
        a=[[0,0],[3,0],[0,3]]
        B=[[[0,0],[0,0]],[[1,0],[0,1]],[[2,1],[1,3]]]
        r=certificate('1/10',a,B,[0,0,0])
        self.assertEqual(r['result'],'conditional_full_null_braid')
        self.assertEqual(set(r['per_pair']),{'0,1','0,2','1,2'})

    def test_source_hessian_to_remainder(self):
        self.assertEqual(taylor_errors_from_hessian([0,1,4],'1/10'),[Fraction(0),Fraction(1,200),Fraction(1,50)])

    def test_nonlinear_remainder_preserves_winding(self):
        a=[[0,0],[2,0]]
        B=[[[0,0],[0,0]],[[1,0],[0,1]]]
        expected=certificate(3,a,B,['1/20',0])
        self.assertEqual(expected['result'],'conditional_affine_braid_preserved')
        actual=pairwise_windings(points(a,B,3,nonlinear=0.01))
        self.assertEqual(actual[(0,1)],expected['per_pair']['0,1']['pair_winding'])

    def test_numeric_polygon_matches_both_radius_regimes(self):
        a=[[0,0],[2,0]];B=[[[0,0],[0,0]],[[1,0],[0,1]]]
        for r in (1,3):
            with self.subTest(r=r):
                self.assertEqual(pairwise_windings(points(a,B,r))[(0,1)],
                                 certificate(r,a,B,[0,0])['per_pair']['0,1']['pair_winding'])

    def test_signs_with_rotated_ellipse(self):
        B=[[[0,0],[0,0]],[[0,1],[-2,0]]]
        self.assertEqual(certificate('3',base=[[0,0],[2,0]],matrices=B)['per_pair']['0,1']['pair_winding'],1)

    def test_remainder_certificate_uncertified_source(self):
        r=certificate(1)
        self.assertFalse(r['source_hessian_verified'])
        self.assertFalse(r['trained_neural_observation_certified'])

    def test_invalid_inputs_refused(self):
        calls=[lambda:certificate(0),lambda:certificate(-1),
               lambda:certificate(1,errors=[-1,0]),
               lambda:certificate(1,errors=[0]),
               lambda:certificate(1,margin=-1),
               lambda:certificate(1,matrices=[[[0,0],[0,0]],[[1,0]]]),
               lambda:certificate(1,base=[[0,0]]),
               lambda:certificate(1,sqrt_digits=30),
               lambda:taylor_errors_from_hessian([1,-1],1),
               lambda:taylor_errors_from_hessian([1],-1)]
        for fn in calls:
            with self.subTest(fn=fn):
                with self.assertRaises((ValueError,TypeError,IndexError)):
                    fn()


class CampaignIntegrityTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.tmp=Path(t.name)
        self.rows={(model,chart):[source_record(k,model,chart) for k in range(4)]
                   for model in MODELS for chart in CHARTS}
        self.paths={(model,chart):self.tmp/(('loop' if model==MODELS[0] else 'ffu')+'_'+chart+'.jsonl')
                    for model in MODELS for chart in CHARTS}
        self.save()

    def save(self):
        for key,rows in self.rows.items():
            self.paths[key].write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))

    def test_complete_synthetic_campaign_passes(self):
        got=strict_compare_files(self.paths,expected_tasks=4)
        self.assertEqual(got['n_matched_tasks'],4)
        self.assertTrue(got['source_type_consistent'])

    def test_within_arm_checkpoint_drift_refused(self):
        self.rows[MODELS[0],CHARTS[0]][1]['checkpoint_sha256']='other'
        self.save()
        with self.assertRaisesRegex(ValueError,'Campaign-wide checkpoint'):
            strict_compare_files(self.paths,expected_tasks=4)

    def test_within_arm_selected_features_drift_refused(self):
        self.rows[MODELS[1],CHARTS[1]][2]['features']=[33,44]
        self.save()
        with self.assertRaisesRegex(ValueError,'Campaign-wide features'):
            strict_compare_files(self.paths,expected_tasks=4)

    def test_source_mixing_refused(self):
        self.rows[MODELS[0],CHARTS[0]][0]['source_type']='hash_verified_RMP_checkpoint'
        self.save()
        with self.assertRaisesRegex(ValueError,'Mixed synthetic'):
            strict_compare_files(self.paths,expected_tasks=4)

    def test_cross_model_token_hash_mismatch_refused(self):
        for chart in CHARTS:
            self.rows[MODELS[1],chart][2]['token_sequence_sha256']='different-source-tokens'
        self.save()
        with self.assertRaisesRegex(ValueError,'Cross-model token sequence'):
            strict_compare_files(self.paths,expected_tasks=4)

    def test_cross_task_readout_chart_drift_refused(self):
        self.rows[MODELS[0],CHARTS[0]][3]['chart_sha256']='different-chart'
        self.save()
        with self.assertRaisesRegex(ValueError,'Campaign-wide chart_sha256'):
            strict_compare_files(self.paths,expected_tasks=4)

    def test_real_format_malformed_hash_refused(self):
        for (model,chart), rows in self.rows.items():
            for row in rows:
                row['source_type']='hash_verified_RMP_checkpoint'
        self.save()
        with self.assertRaisesRegex(ValueError,'malformed'):
            strict_compare_files(self.paths,expected_tasks=4)

    def test_full_mock_real_shape_passes_format_only(self):
        # Artificial records exercise format validation. These are TEST MOCKS,
        # not observations from a trained RMP checkpoint.
        for (model,chart), rows in self.rows.items():
            for k,row in enumerate(rows):
                row['source_type']='hash_verified_RMP_checkpoint'
                for key in ('checkpoint_sha256','dictionary_sha256','intervention_axes_sha256','chart_sha256'):
                    import hashlib
                    row[key]=hashlib.sha256((model+chart+key if key=='chart_sha256' else model+key).encode()).hexdigest()
                for key in ('task_id_sha256','group_id_sha256','token_sequence_sha256'):
                    import hashlib
                    row[key]=hashlib.sha256((str(k)+key).encode()).hexdigest()
                row['capture']={
                    'angles':[2*math.pi*i/32 for i in range(33)],
                    'predictions':[0]*33,'baseline_prediction':0,
                    'frames_by_visit':[[[[float(j*3),float(j*3+1)] for j in range(3)] for _ in range(33)] for _ in range(8)],
                    'injected_position':4,'injected_after_visit':0,
                    'observed_token_positions':[16,32,48],
                    'radius_in_token_rms':0.05,
                    'changed_prediction_fraction':0.0,
                    'max_abs_logit_change_from_clean':0.,
                    'chart_sha256':row['chart_sha256'],
                    'intervention_axes_sha256':row['intervention_axes_sha256'],
                }
                row['topology']=[{'valid':True,'artin':{'status':'exact','nonidentity':False}} for _ in range(8)]
                row['double_resolution']=[{'match':True} for _ in range(8)]
        self.save()
        got=strict_compare_files(self.paths,expected_tasks=4)
        self.assertEqual(got['source_classification'],'real_RMP_capture_uncertified_floating_topology')
        self.assertFalse(got['neural_causal_claim_certified'])
        # Malformed pair frames must not silently pass.
        self.rows[MODELS[0],CHARTS[0]][0]['capture']['frames_by_visit'][0][0][0][0]=float('nan')
        self.save()
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            strict_compare_files(self.paths,expected_tasks=4)


class SampledDiagnosticTests(unittest.TestCase):
    def test_known_affine_radius_transition_is_only_empirical(self):
        a=[[0,0],[2,0]]
        B=[[[0,0],[0,0]],[[1,0],[0,1]]]
        output=screen_frames(points(a,B,3,64))
        self.assertEqual(output['sampled_agreement']['0,1'],True)
        self.assertEqual(output['polygonal_pair_windings']['0,1'],1)
        self.assertEqual(output['status'],'empirical_only_unverified_full_circle_error')
        self.assertFalse(output['source_hessian_certified'])

    def test_four_turn_alias_not_promoted_to_certificate(self):
        # Four turns sampled at the same four angular positions can alias to zero.
        frames=[]
        for k in range(9):
            t=2*math.pi*k/8
            frames.append([(0.,0.),(2*math.cos(4*t),2*math.sin(4*t))])
        output=screen_frames(frames)
        self.assertEqual(output['status'],'empirical_only_unverified_full_circle_error')
        self.assertFalse(output['source_hessian_certified'])

    def test_braid_commutator_keeps_empirical_scope(self):
        c=synthetic_braid(3,[1,1,2,2,-1,-1,-2,-2])
        out=screen_frames(c)
        self.assertEqual(out['polygonal_pair_windings'],{'0,1':0,'0,2':0,'1,2':0})
        self.assertEqual(out['status'],'empirical_only_unverified_full_circle_error')

    def test_invalid_sampled_closure_refused(self):
        a=[[0,0],[2,0]];B=[[[0,0],[0,0]],[[1,0],[0,1]]]
        v=points(a,B,2,32)
        v[-1][0]=(1.,1.)
        with self.assertRaisesRegex(ValueError,'closure'):
            screen_frames(v)

    def test_quadratic_taylor_remainder_exact_upper(self):
        # For F(u,v) = A + B(u,v) + (h/2)(u^2-v^2,2uv),
        # ||D^2 F||op = |h| and ||remainder|| = |h|r^2/2.
        radius=0.1;h=3.0
        bound=taylor_errors_from_hessian(['3'],'1/10')[0]
        for k in range(100):
            t=2*math.pi*k/100
            u,v=radius*math.cos(t),radius*math.sin(t)
            d=(h/2*(u*u-v*v),h*u*v)
            self.assertLessEqual(math.hypot(*d),float(bound)+1e-14)

    def test_readonly_capture_receipt_smoke(self):
        a=[[0,0],[2,0]];B=[[[0,0],[0,0]],[[1,0],[0,1]]]
        capture={'task_id_sha256':'synthetic-task','model':'synthetic_model',
                 'source_type':'synthetic_fixture','capture':{'frames_by_visit':[points(a,B,3,32)]}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'in.jsonl';out=Path(tmp)/'out.json'
            p.write_text(json.dumps(capture)+'\n')
            self.assertEqual(audit_capture(p,out),1)
            record=json.loads(out.read_text())
            self.assertEqual(record['rows'][0]['screen_per_visit'][0]['status'],
                             'empirical_only_unverified_full_circle_error')
            with self.assertRaises(FileExistsError):audit_capture(p,out)


if __name__=='__main__':
    unittest.main()
