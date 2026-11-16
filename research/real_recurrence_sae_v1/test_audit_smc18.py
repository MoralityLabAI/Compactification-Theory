"""SMC-18 independent numerical receipt replay mutation regressions."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_smc18 as a


class SMC18AuditTests(unittest.TestCase):
    def toy_rows(self):
        out=[]
        for p in range(64):
            active=p<24
            metric={'finite_kill_minus_sham':2.,'clean_linear_prediction':1.8}
            out.append({'eligible_mask':[active]+[False]*7,
                        'metrics':[copy.deepcopy(metric) if active else None]+[None]*7})
        return out

    def test_replay_registered_puzzle_bootstrap_and_sign_gate(self):
        result=a.independent_score(self.toy_rows(),seed=202618130)
        self.assertEqual(result['eligible_goals'],24)
        self.assertEqual(result['eligible_puzzles'],24)
        self.assertEqual(result['comparable_goals'],24)
        self.assertTrue(result['gate'])
        self.assertEqual(result['puzzle_bootstrap95_finite'],[2.,2.])
        self.assertAlmostEqual(result['relative_absolute_remainder'],.1)

    def test_forged_negative_gradient_residual_fails_registered_gate(self):
        rows=self.toy_rows()
        for r in rows:
            if r['metrics'][0] is not None:
                r['metrics'][0]['clean_linear_prediction']=-1.
        measured=a.independent_score(rows,seed=202618130)
        self.assertFalse(measured['registered_gradient_sign_gate'])
        self.assertFalse(measured['registered_linear_remainder_gate'])
        self.assertFalse(measured['gate'])

    def toy_dual(self):
        cov=[[0.]*48 for _ in range(96)]
        for i in range(48):cov[i+4][i]=1.
        g=[0.]*48;g[0]=2.;g[1]=-3.
        k=[0.]*48;k[0]=1.
        s=[0.]*48
        lam=[0.]*92;lam[0]=2.;lam[1]=-3.
        return {'gradient_48':g,'kill_delta_48':k,'sham_delta_48':s,
                'active_feature_ids':[0,1,2,3],
                'inactive_gradient_dual':{
                  'minimum_inactive_singular_value':1.,
                  'full_inactive_rank':48,
                  'lambda_minimum_norm':lam,
                  'gradient_reconstructed_from_inactive':g[:],
                  'reconstruction_l2_error':0.,
                  'inactive_shift_l2':1.,
                  'task_gradient_dual_dot':2.,
                  'task_gradient_inactive_cs_ceiling':13**.5}},cov

    def test_independent_cholesky_reconstructs_minimal_dual(self):
        metric,cov=self.toy_dual()
        a.audit_inactive_dual(metric,cov)

    def test_a_nullspace_shift_forgery_cannot_be_called_minimum_norm(self):
        metric,cov=self.toy_dual()
        wrong=copy.deepcopy(metric)
        wrong['inactive_gradient_dual']['lambda_minimum_norm'][-1]=100.
        with self.assertRaisesRegex(ValueError,'minimum-norm'):
            a.audit_inactive_dual(wrong,cov)

    def test_forged_task_directional_derivative_detected(self):
        metric,cov=self.toy_dual()
        wrong=copy.deepcopy(metric)
        wrong['inactive_gradient_dual']['task_gradient_dual_dot']=3
        with self.assertRaisesRegex(ValueError,'attribution'):
            a.audit_inactive_dual(wrong,cov)

    def test_schema_and_preregistration_sha_guarded(self):
        with self.assertRaisesRegex(ValueError,'format'):
            a.audit({'schema':'wrong'})
        with self.assertRaisesRegex(ValueError,'preregistration'):
            a.audit({'schema':'SMC-18/task-gradient-inactive-dual-v1',
                     'pre_registration_commit':'0000000000000000000000000000000000000000'})


if __name__=='__main__':unittest.main()
