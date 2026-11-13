"""SCM-07 exact norm inequalities and real-analytic mock RMP differential tests."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from local_jacobian import (centered_norm_sq, local_layernorm_lipschitz,
                           block_token_enclosure, capture_float_clean_centers,
                           visit_token_bounds)
from rmp_lipschitz import visit_bounds
from scm07_audit import audit_trace, audit_jsonl
from test_rmp_lipschitz import MiniModel, MiniExplicitBlock


class ExactLocalLayerNorm(unittest.TestCase):
    def test_dyadic_center_energy_exact(self):
        self.assertEqual(centered_norm_sq(torch.tensor([1.,2.,3.,4.])),Fraction(5))

    def test_invariant_to_shared_shift(self):
        norm=nn.LayerNorm(4)
        x=torch.tensor([1.,2.,4.,8.])
        y=x+16.
        self.assertEqual(centered_norm_sq(x),centered_norm_sq(y))
        self.assertEqual(local_layernorm_lipschitz(norm,x,'.01'),
                         local_layernorm_lipschitz(norm,y,'.01'))

    def test_high_variance_improves_global(self):
        norm=nn.LayerNorm(4)
        from rmp_lipschitz import layernorm_bounds
        x=torch.tensor([-3.,-1.,1.,3.])
        self.assertLess(local_layernorm_lipschitz(norm,x,Fraction(1,100)),
                        layernorm_bounds(norm)[0]/100)

    def test_nearly_constant_reverts_to_global(self):
        norm=nn.LayerNorm(4)
        x=torch.ones(4)
        from rmp_lipschitz import layernorm_bounds
        self.assertEqual(local_layernorm_lipschitz(norm,x,.01),layernorm_bounds(norm)[0])

    def test_large_error_surrenders_improvement(self):
        norm=nn.LayerNorm(4)
        x=torch.tensor([-3.,-1.,1.,3.])
        from rmp_lipschitz import layernorm_bounds
        self.assertEqual(local_layernorm_lipschitz(norm,x,.01,center_error=100),layernorm_bounds(norm)[0])

    def test_local_jacobians_against_autodiff(self):
        norm=nn.LayerNorm(4).double()
        center=torch.tensor([-2.,-.5,1.,2.],dtype=torch.float64)
        radius=Fraction(1,4)
        Lip=float(local_layernorm_lipschitz(norm,center,radius))
        for j in range(12):
            offset=torch.tensor([math.sin(j+2),math.cos(j),math.sin(3*j),math.cos(2*j)],dtype=torch.float64)
            offset=offset/offset.norm()*(j/12*.25)
            jac=torch.autograd.functional.jacobian(norm,center+offset)
            self.assertLessEqual(float(torch.linalg.matrix_norm(jac,ord=2)),Lip+1e-9)

    def test_negative_or_wrong_dimension_refused(self):
        norm=nn.LayerNorm(4)
        with self.assertRaises(ValueError):
            local_layernorm_lipschitz(norm,torch.ones(4),-1)
        with self.assertRaises(ValueError):
            local_layernorm_lipschitz(norm,torch.ones(3),0)
        with self.assertRaises(ValueError):
            local_layernorm_lipschitz(norm,torch.ones(4),0,center_error=-1)

    def test_nonfinite_center_refused(self):
        with self.assertRaises(ValueError):
            centered_norm_sq(torch.tensor([1.,float('nan')]))


class CausalBlockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(13)
        cls.model=MiniModel(tied=True,beta=.1).eval()
        cls.input=torch.randn(1,5,4)
        cls.axes=torch.eye(4)[:,:2]
        cls.chart=cls.axes

    def test_causal_frontier_preserves_exact_zeros(self):
        block=self.model.blocks[0]
        r=[Fraction(0),Fraction(0),Fraction(1,50),Fraction(0),Fraction(0)]
        out=block_token_enclosure(block,r)['output_radii']
        self.assertEqual(out[:2],(0,0))
        self.assertGreater(out[2],0)
        self.assertGreater(out[4],0)

    def test_center_free_tighter_than_previous_global(self):
        a=visit_bounds(self.model,sequence=5,chart=self.chart,axes=self.axes,
                       amplitude=.02,observed_tokens=(1,2,3),injected_token=0)
        b=visit_token_bounds(self.model,sequence=5,chart=self.chart,axes=self.axes,
                       amplitude=.02,observed_tokens=(1,2,3),injected_token=0)
        self.assertLess(Fraction(b['per_visit_pair_bounds'][-1]['0,1']),
                        Fraction(a['per_visit_pair_bounds'][-1]['0,1']))
        self.assertEqual(b['bound_scope'],'source_global_unconditional')

    def test_conditional_local_stronger(self):
        c=capture_float_clean_centers(self.model,self.input)
        e=[[0]*5 for _ in range(3)]
        a=visit_token_bounds(self.model,sequence=5,chart=self.chart,axes=self.axes,
                       amplitude=.02,observed_tokens=(1,2,3),injected_token=0)
        b=visit_token_bounds(self.model,sequence=5,chart=self.chart,axes=self.axes,
                       amplitude=.02,observed_tokens=(1,2,3),injected_token=0,
                       pre_centers=c['pre'],mid_centers=c['mid'],
                       pre_center_errors=e,mid_center_errors=e)
        self.assertLess(Fraction(b['per_visit_pair_bounds'][-1]['0,1']),
                        Fraction(a['per_visit_pair_bounds'][-1]['0,1']))
        self.assertIn('conditional',b['bound_scope'])
        self.assertFalse(b['center_enclosures_verified_by_this_program'])
        self.assertEqual(c['source'],'uncertified_float_PyTorch_forward')

    def test_jvp_global_tied_and_untied(self):
        for tied in (True,False):
            torch.manual_seed(17)
            model=MiniModel(tied=tied,beta=.1).eval()
            bounds=visit_token_bounds(model,sequence=5,chart=self.chart,axes=self.axes,
                  amplitude=.025,observed_tokens=(1,2,3),injected_token=0)
            for theta in (-1.5,0.2,2.4):
                t=torch.tensor(theta,requires_grad=True)
                for visit in (1,2):
                    def q(a):
                        z=model.traces(self.input,self.axes,self.chart,.025,a)[visit]
                        return z[0]-z[1]
                    jac=torch.autograd.functional.jvp(q,(t,),(torch.ones_like(t),))[1]
                    self.assertLessEqual(float(jac.norm()),
                         float(Fraction(bounds['per_visit_pair_bounds'][visit]['0,1']))+1e-6)

    def test_jvp_local_is_empirical_check_only(self):
        c=capture_float_clean_centers(self.model,self.input)
        e=[[0]*5 for _ in range(3)]
        bounds=visit_token_bounds(self.model,sequence=5,chart=self.chart,axes=self.axes,
                  amplitude=.025,observed_tokens=(1,2,3),injected_token=0,
                  pre_centers=c['pre'],mid_centers=c['mid'],
                  pre_center_errors=e,mid_center_errors=e)
        for theta in (0.,.4,1.6,2.8):
            t=torch.tensor(theta,requires_grad=True)
            for visit in (1,2):
                def q(a):
                    z=self.model.traces(self.input,self.axes,self.chart,.025,a)[visit]
                    return z[0]-z[1]
                jac=torch.autograd.functional.jvp(q,(t,),(torch.ones_like(t),))[1]
                self.assertLessEqual(float(jac.norm()),
                      float(Fraction(bounds['per_visit_pair_bounds'][visit]['0,1']))+1e-7)

    def test_two_point_tube_bound_empirical(self):
        block=self.model.blocks[0]
        X=self.input.clone()
        # Explicitly collect clean mid and pre at one block.
        inner=[]
        hook=block.norm2.register_forward_pre_hook(lambda _m,args:inner.append(args[0][0].detach().clone()))
        try:
            baseline=block(X)
        finally:
            hook.remove()
        r=[Fraction(0)]*5
        r[2]=Fraction(1,100)
        E=block_token_enclosure(block,r,pre_center=X[0],mid_center=inner[0],
               pre_center_errors=[0]*5,mid_center_errors=[0]*5)['output_radii']
        X1=X.clone()
        X1[0,2]+=torch.tensor([.005,-.005,.005,-.005])
        Y1=block(X1)
        for j in range(5):
            self.assertLessEqual(float((Y1[0,j]-baseline[0,j]).norm().detach()),float(E[j])+1e-6)

    def test_beta_zero_keeps_unobserved_tokens_fixed(self):
        model=MiniModel(tied=False,beta=0).eval()
        b=visit_token_bounds(model,sequence=5,chart=self.chart,axes=self.axes,
                   amplitude=.25,observed_tokens=(1,2,4),injected_token=0)
        self.assertTrue(all(all(v=='0' for v in row.values()) for row in b['per_visit_pair_bounds']))

    def test_prefix_zero_for_all_visits(self):
        model=MiniModel(tied=False).eval()
        b=visit_token_bounds(model,sequence=5,chart=self.chart,axes=self.axes,
                   amplitude=1.,observed_tokens=(0,1,2),injected_token=4)
        self.assertTrue(all(all(v=='0' for v in row.values()) for row in b['per_visit_pair_bounds']))

    def test_model_without_eval_refused(self):
        with self.assertRaisesRegex(ValueError,'eval'):
            visit_token_bounds(MiniModel(),sequence=5,chart=self.chart,axes=self.axes,
                  amplitude=.02,observed_tokens=(1,2,3),injected_token=0)

    def test_required_centers_without_errors_refused(self):
        c=capture_float_clean_centers(self.model,self.input)
        with self.assertRaisesRegex(ValueError,'require per-token certified-error claims'):
            block_token_enclosure(self.model.blocks[0],[0]*5,pre_center=c['pre'][0])

    def test_unsupported_approximate_gelu_refused(self):
        m=MiniExplicitBlock()
        m.mlp[1]=nn.GELU(approximate='tanh')
        with self.assertRaisesRegex(ValueError,'exact erf GELU'):
            block_token_enclosure(m,[.01]*5)

    def test_invalid_axis_dim_refused(self):
        with self.assertRaises(ValueError):
            visit_token_bounds(self.model,sequence=5,chart=torch.eye(3)[:,:2],axes=torch.eye(3)[:,:2],
                  amplitude=.02,observed_tokens=(1,2,3),injected_token=0)


class AuditControls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.m=MiniModel(tied=True).eval()
        self.chart=torch.eye(4)[:,:2]
        self.bounds=visit_token_bounds(self.m,sequence=5,chart=self.chart,axes=self.chart,
                 amplitude='.0000001',observed_tokens=(1,2,3),injected_token=0)

    def test_conditional_audit_pass_does_not_certify_float_neural_link(self):
        N=16
        pts=[[0.,0.],[1.,0.],[0.,1.]]
        trace={'angles':[2*math.pi*k/N for k in range(N+1)],
               'frames_by_visit':[[pts for _ in range(N+1)] for _ in range(3)]}
        report=audit_trace(trace,self.bounds)
        self.assertEqual(report['source_bound_scope'],'source_global_unconditional')
        self.assertTrue(all(v['passes_ideal_sampling_inequality'] for v in report['visits']))
        self.assertFalse(report['observed_link_certified'])

    def test_nonclosed_trace_refused_by_audit(self):
        N=16
        pts=[[0.,0.],[1.,0.],[0.,1.]]
        shifted=[[0.,0.],[.9,0.],[0.,1.]]
        trace={'angles':[2*math.pi*k/N for k in range(N+1)],
               'frames_by_visit':[[pts for _ in range(N)]+[shifted] for _ in range(3)]}
        result=audit_trace(trace,self.bounds)
        self.assertFalse(result['visits'][0]['passes_ideal_sampling_inequality'])

    def test_nonuniform_angles_refused(self):
        N=16; angles=[2*math.pi*k/N for k in range(N+1)];angles[2]+=.03
        trace={'angles':angles,'frames_by_visit':[[[[0.,0.],[1.,0.],[0.,1.]] for _ in range(N+1)] for _ in range(3)]}
        with self.assertRaisesRegex(ValueError,'Nonuniform'):
            audit_trace(trace,self.bounds)

    def test_modified_bound_schema_refused(self):
        with self.assertRaisesRegex(ValueError,'schema'):
            audit_trace({'frames_by_visit':[]},{'schema':'fake'})

    def test_audit_refuses_replacing_an_existing_receipt(self):
        from scm07_audit import audit_jsonl
        import tempfile, json
        N=8
        pts=[[0.,0.],[1.,0.],[0.,1.]]
        trace={'angles':[2*math.pi*k/N for k in range(N+1)],
               'frames_by_visit':[[pts for _ in range(N+1)] for _ in range(3)]}
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'capture.jsonl'; dst=Path(d)/'audit.jsonl'
            src.write_text(json.dumps({'capture':trace})+'\n')
            audit_jsonl(src,dst,bounds=self.bounds)
            with self.assertRaises(FileExistsError):
                audit_jsonl(src,dst,bounds=self.bounds)

    def test_reference_fixture_passes_radius_threshold(self):
        from scm07_reference import run
        result=run()
        self.assertEqual(result['schema'],'scm07_cpu_reference_v1')
        self.assertTrue(all(result['radii']['1/1000000']['disk_inequality_conditionally_passes']))
        self.assertFalse(result['radii']['1/10000']['disk_inequality_conditionally_passes'][-1])
        self.assertFalse(result['real_activation_certified'])
        self.assertTrue(result['strict_causal_prefix_injected_at_63'])

    def test_local_refuses_center_error_length_mismatch(self):
        block=self.m.blocks[0]
        with self.assertRaisesRegex(ValueError,'per-token'):
            block_token_enclosure(block,[Fraction(0)]*5,
                      pre_center=torch.zeros(5,4),pre_center_errors=[0]*4)

    def test_local_audit_keeps_center_assumption_visible(self):
        N=8
        pts=[[0.,0.],[1.,0.],[0.,1.]]
        trace={'angles':[2*math.pi*k/N for k in range(N+1)],
               'frames_by_visit':[[pts for _ in range(N+1)] for _ in range(3)]}
        altered=dict(self.bounds,bound_scope='conditional_on_external_ideal_center_enclosures')
        result=audit_trace(trace,altered)
        self.assertFalse(result['source_global_derivative_proved'])
        self.assertFalse(result['ideal_center_enclosures_proved'])


if __name__=='__main__':
    unittest.main()
