"""CPU regression for exact stored-weight real arithmetic bounds.

MiniExplicitBlock reproduces the RMP ExplicitDecoderBlock computation and
causal attention layout. Actual checkpoint-backed execution remains a separate
gate because its artifacts are local to the original RMP worker.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rmp_lipschitz import (spectral_upper, sqrt_upper, sqrt_lower, ceiling_decimal,
       layernorm_bounds, block_bound, visit_bounds, topology_zero_disk_certificate,
       angular_gap_upper)
from scm06_audit import examine_record
from rmp_capture import capture_cycle


class MiniExplicitBlock(nn.Module):
    def __init__(self, d=4, h=2, alpha=1.0, beta=.1):
        super().__init__()
        self.hidden_size=d
        self.num_heads=h
        self.head_size=d//h
        self.alpha=alpha
        self.beta=beta
        self.norm1=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False)
        self.attention_out=nn.Linear(d,d,bias=False)
        self.norm2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,4*d,bias=False),nn.GELU(),nn.Linear(4*d,d,bias=False))
        with torch.no_grad():
            self.qkv.weight.mul_(.08)
            self.attention_out.weight.mul_(.08)
            self.mlp[0].weight.mul_(.08)
            self.mlp[2].weight.mul_(.08)

    def forward(self,x):
        B,T,D=x.shape
        heads=self.num_heads
        qkv=self.qkv(self.norm1(x)).reshape(B,T,3,heads,D//heads)
        Q,K,V=(part.transpose(1,2) for part in qkv.unbind(dim=2))
        logits=(Q@K.transpose(-2,-1))/(D//heads)**.5
        mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),diagonal=1)
        p=torch.softmax(logits.masked_fill(mask,float('-inf')),dim=-1)
        hv=p@V
        attended=self.attention_out(hv.transpose(1,2).reshape(B,T,D))
        middle=self.alpha*x+self.beta*attended
        return self.alpha*middle+self.beta*self.mlp(self.norm2(middle))


class MiniModel(nn.Module):
    def __init__(self, tied=True, beta=.1):
        super().__init__()
        self.blocks=nn.ModuleList([MiniExplicitBlock(beta=beta)] if tied else [MiniExplicitBlock(beta=beta) for _ in range(3)])
        self.module_indices=(0,0,0) if tied else (0,1,2)

    def traces(self, initial, axes, chart, amplitude, theta):
        out=[]
        x=initial
        for visit,index in enumerate(self.module_indices):
            x=self.blocks[index](x)
            if visit==0:
                shifted=x.clone()
                shifted[0,0]=shifted[0,0]+amplitude*(axes[:,0]*torch.cos(theta)+axes[:,1]*torch.sin(theta))
                x=shifted
            out.append((x[0,1:4]@chart))
        return out


class MatrixTests(unittest.TestCase):
    def test_exact_spectral_norm_diagonal(self):
        a=torch.tensor([[3.,0.],[0.,4.]])
        self.assertEqual(spectral_upper(a),4)

    def test_spectral_bound_is_above_true_svd(self):
        g=torch.Generator().manual_seed(20261113)
        for shape in ((1,4),(2,2),(8,12),(12,8),(5,1)):
            A=torch.randn(*shape,generator=g)
            self.assertGreaterEqual(float(spectral_upper(A))+1e-12,
                                    float(torch.linalg.matrix_norm(A.double(),ord=2)))

    def test_subnormal_float32_exact(self):
        small=torch.tensor([[1e-44,0.]],dtype=torch.float32)
        self.assertEqual(spectral_upper(small),Fraction(*float(small[0,0]).as_integer_ratio()))

    def test_nan_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            spectral_upper(torch.tensor([[float('nan')]]))

    def test_exact_sqrt_outwards(self):
        self.assertLessEqual(sqrt_lower(Fraction(2))**2,2)
        self.assertGreaterEqual(sqrt_upper(Fraction(2))**2,2)
        self.assertEqual(sqrt_upper(Fraction(9)),3)
        self.assertEqual(ceiling_decimal(Fraction(7,3),digits=1),Fraction(24,10))

    def test_layernorm_bounds_random_jacobian(self):
        norm=nn.LayerNorm(4)
        norm.bias.data=torch.tensor([.1,-.2,.3,.1])
        Lip,M=layernorm_bounds(norm)
        for offset in (0.,1e-3,5.):
            x=(torch.tensor([offset,offset+1e-3,offset-1e-3,offset],dtype=torch.float64))
            norm=norm.to(torch.float64)
            jac=torch.autograd.functional.jacobian(norm,x)
            self.assertLessEqual(float(torch.linalg.matrix_norm(jac,ord=2)),float(Lip)+1e-8)
            self.assertLessEqual(float(torch.linalg.vector_norm(norm(x))),float(M)+1e-8)

    def test_interval_bound_uses_true_circle(self):
        for n in (8,16,64):
            self.assertGreater(float(angular_gap_upper(n)),2*math.pi/n)


class BlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(3)
        cls.model=MiniModel(tied=True).eval()
        cls.axes=torch.eye(4)[:,:2].contiguous()
        cls.chart=torch.stack([torch.tensor([1.,0.,0.,0.]),torch.tensor([0.,1.,0.,0.])],dim=1)
        cls.initial=torch.randn(1,5,4)

    def test_global_block_bound_across_random_pairs(self):
        bound=block_bound(self.model.blocks[0],sequence=5)
        self.assertGreater(bound['full'],1)
        self.assertGreater(bound['attention'],0)
        torch.manual_seed(9)
        for _ in range(10):
            X=torch.randn(1,5,4)
            Y=torch.randn(1,5,4)
            measured=float((self.model.blocks[0](X)-self.model.blocks[0](Y)).norm().detach())
            allowed=float(bound['full'])*float((X-Y).norm())
            self.assertLessEqual(measured,allowed+1e-8)

    def test_jvp_at_every_visit_tied(self):
        bounds=visit_bounds(self.model, sequence=5,chart=self.chart,axes=self.axes,
                 amplitude=Fraction(1,50),observed_tokens=(1,2,3),injected_token=0)
        self.assertEqual(len(bounds['per_visit_pair_bounds']),3)
        self.assertTrue(all(value=='0' for value in bounds['per_visit_pair_bounds'][0].values()))
        for theta in (-2.3,-.1,0.3,2.7):
            x=torch.tensor(theta,dtype=torch.float32,requires_grad=True)
            for visit in range(3):
                def q(xx,visit=visit):
                    z=self.model.traces(self.initial,self.axes,self.chart,.02,xx)[visit]
                    return z[0]-z[1]
                derivative=torch.autograd.functional.jvp(q,(x,),(torch.ones_like(x),))[1]
                bound=float(Fraction(bounds['per_visit_pair_bounds'][visit]['0,1']))
                self.assertLessEqual(float(torch.linalg.vector_norm(derivative)),bound+1e-5)

    def test_jvp_at_every_visit_untied(self):
        torch.manual_seed(5)
        model=MiniModel(tied=False).eval()
        bounds=visit_bounds(model,sequence=5,chart=self.chart,axes=self.axes,
                amplitude=.04,observed_tokens=(1,2,3),injected_token=0)
        self.assertEqual([x['module_index'] for x in bounds['per_block_factors']],[1,2])
        for theta in (0.,1.1,2.6):
            x=torch.tensor(theta,requires_grad=True)
            for visit in (1,2):
                def q(xx,visit=visit):
                    z=model.traces(self.initial,self.axes,self.chart,.04,xx)[visit]
                    return z[0]-z[2]
                derivative=torch.autograd.functional.jvp(q,(x,),(torch.ones_like(x),))[1]
                bound=float(Fraction(bounds['per_visit_pair_bounds'][visit]['0,2']))
                self.assertLess(float(derivative.norm()),bound)

    def test_unsupported_gelu_refused(self):
        block=MiniExplicitBlock()
        block.mlp[1]=nn.GELU(approximate='tanh')
        with self.assertRaisesRegex(ValueError,'exact erf GELU'):
            block_bound(block,sequence=5)

    def test_unsupported_qkv_bias_refused(self):
        block=MiniExplicitBlock()
        block.qkv=nn.Linear(4,12,bias=True)
        with self.assertRaisesRegex(ValueError,'Biased QKV'):
            block_bound(block,sequence=5)

    def test_training_model_refused(self):
        m=MiniModel()
        with self.assertRaisesRegex(ValueError,'eval'):
            visit_bounds(m,sequence=5,chart=self.chart,axes=self.axes,
                         amplitude=1.,observed_tokens=(1,2,3),injected_token=0)

    def test_disk_null_at_tiny_radius(self):
        torch.manual_seed(0)
        m=MiniModel(tied=False,beta=0).eval()
        chart=self.chart
        axes=self.axes
        b=visit_bounds(m,sequence=5,chart=chart,axes=axes,
                       amplitude=Fraction(1,10**9),observed_tokens=(1,2,3),injected_token=0)
        points=[[[0.,0.],[10.,0.],[0.,10.]] for _ in range(3)]
        disk=topology_zero_disk_certificate(points,pair_bounds_result=b,margin='0')
        self.assertTrue(all(x['conditional_null_if_clean_baseline_exact'] for x in disk))

    def test_disk_refuses_projected_collision(self):
        b=visit_bounds(self.model,sequence=5,chart=self.chart,axes=self.axes,
                       amplitude=1,observed_tokens=(1,2,3),injected_token=0)
        points=[[[0.,0.],[0.,0.],[1.,0.]] for _ in range(3)]
        disk=topology_zero_disk_certificate(points,pair_bounds_result=b)
        self.assertFalse(disk[0]['conditional_null_if_clean_baseline_exact'])

    def test_audit_refuses_legacy_capture_without_scale(self):
        record={'capture':{'frames_by_visit':[],'angles':[]}}
        result=examine_record(record,SimpleNamespace(model=self.model),self.chart,self.axes)
        self.assertEqual(result['status'],'missing_new_capture_fields')
        self.assertFalse(result['machine_level_certified'])

    def test_analytic_audit_passes_visit_zero_only(self):
        N=16
        with torch.no_grad():
            base=[]; traces=[]
            # Visit zero has no perturbation at observed positions. Constant
            # chart arrays are also used here to exercise the conservative
            # derivative-certification status, not to fabricate model evidence.
            for k in range(3):
                pts=[[0.,0.],[10.,0.],[0.,10.]]
                base.append(pts)
                traces.append([pts for _ in range(N+1)])
        rec={'model':'mock','capture':{'angles':[2*math.pi*k/N for k in range(N+1)],
             'frames_by_visit':traces,'clean_chart_by_visit':base,
             'injection_baseline_rms':1.,'sequence_length':5,
             'radius_in_token_rms':1e-8,'observed_token_positions':[1,2,3],
             'injected_position':0,'injected_after_visit':0}}
        report=examine_record(rec,SimpleNamespace(model=self.model),self.chart,self.axes)
        self.assertEqual(report['status'],'analysed')
        self.assertTrue(report['visits'][0]['sampling_inequality_passed'])
        self.assertFalse(report['visits'][0]['real_model_certified'])
        self.assertFalse(report['machine_level_certified'])


if __name__=='__main__':
    unittest.main()

class CausalAndProvenanceTests(unittest.TestCase):
    def test_strict_causal_prefix_controls_have_zero_lipschitz(self):
        torch.manual_seed(3)
        m=MiniModel(tied=True).eval()
        chart=torch.eye(4)[:,:2]
        observed=(0,1,3)
        # Perturbation at position 4, all three observed positions precede it.
        b=visit_bounds(m,sequence=5,chart=chart,axes=chart,amplitude=1.,
                 observed_tokens=observed,injected_token=4)
        self.assertTrue(all(all(v=='0' for v in row.values())
                            for row in b['per_visit_pair_bounds']))

    def test_mixed_prefix_pair_stays_zero(self):
        m=MiniModel().eval()
        chart=torch.eye(4)[:,:2]
        b=visit_bounds(m,sequence=5,chart=chart,axes=chart,amplitude=1.,
                       observed_tokens=(0,1,3),injected_token=2)
        self.assertEqual([v['0,1'] for v in b['per_visit_pair_bounds']],['0','0','0'])
        self.assertGreater(Fraction(b['per_visit_pair_bounds'][-1]['0,2']),0)

    def test_audited_code_pin_rejects_modified_decoder(self):
        import tempfile
        from scm06_audit import verify_rmp_source, PINNED_RMP_SOURCE
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            for relative in PINNED_RMP_SOURCE:
                f=base/relative
                f.parent.mkdir(parents=True,exist_ok=True)
                f.write_text('mock substituted function()\n')
            with self.assertRaisesRegex(ValueError,'source hash changed'):
                verify_rmp_source(base)

class FullAuditAndCaptureControls(unittest.TestCase):
    def test_source_pin_accepts_matching_blob(self):
        import hashlib
        import tempfile
        from scm06_audit import PINNED_RMP_SOURCE, verify_rmp_source
        raw=b'known minimal local module\n'
        blob=b'blob '+str(len(raw)).encode()+b'\x00'+raw
        sha=hashlib.sha1(blob).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            for path in PINNED_RMP_SOURCE:
                file=base/path
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_bytes(raw)
            previous=dict(PINNED_RMP_SOURCE)
            try:
                for key in PINNED_RMP_SOURCE:PINNED_RMP_SOURCE[key]=sha
                self.assertEqual(set(verify_rmp_source(base).values()),{sha})
            finally:
                PINNED_RMP_SOURCE.clear()
                PINNED_RMP_SOURCE.update(previous)

    def test_new_capture_retains_baseline_scale_and_clean_chart(self):
        from test_rmp_capture import MockRMP
        tokens=torch.tensor([[0,1,2,3]])
        trace=capture_cycle(MockRMP(),tokens,torch.eye(2),
                       inject_token=0,observed_tokens=[1,2],radius=.1,angles=16)
        self.assertEqual(trace['sequence_length'],4)
        self.assertEqual(trace['injection_baseline_rms'],1.)
        self.assertEqual(len(trace['clean_chart_by_visit']),4)
        self.assertEqual(trace['clean_chart_by_visit'][0],[[0.,0.],[.4000000059604645,0.]])

    def test_real_topology_inconsistency_alarm(self):
        m=MiniModel(tied=True).eval()
        chart=torch.eye(4)[:,:2]
        N=16
        frame=[[0.,0.],[100.,0.],[0.,100.]]
        record={'model':'adversarial_mock',
           'capture':{
             'angles':[2*math.pi*k/N for k in range(N+1)],
             'frames_by_visit':[[frame for _ in range(N+1)] for _ in range(3)],
             'clean_chart_by_visit':[frame for _ in range(3)],
             'injection_baseline_rms':1.,'sequence_length':5,
             'radius_in_token_rms':1e-9,'observed_token_positions':[0,1,3],
             'injected_position':4,'injected_after_visit':0},
           'topology':[{'valid':True,'pair_winding':{'0,1':1},
                        'artin':{'status':'exact','nonidentity':True}}]*3}
        out=examine_record(record,SimpleNamespace(model=m),chart,chart)
        self.assertTrue(out['visits'][0]['sampling_inequality_passed'])
        self.assertFalse(out['visits'][0]['real_model_certified'])
        self.assertEqual(out['visits'][0]['consistency_alarm'],'conditional_ideal_topology_inconsistency')

    def test_wrong_uniform_angles_refused(self):
        m=MiniModel().eval()
        chart=torch.eye(4)[:,:2]
        N=16
        frame=[[0.,0.],[100.,0.],[0.,100.]]
        angles=[2*math.pi*k/N for k in range(N+1)]
        angles[5]+=.001
        record={'capture':{'angles':angles,'frames_by_visit':[[frame]*(N+1)]*3,
               'clean_chart_by_visit':[frame]*3,'injection_baseline_rms':1.,
               'sequence_length':5,'radius_in_token_rms':.001,
               'observed_token_positions':[1,2,3],
               'injected_position':0,'injected_after_visit':0}}
        with self.assertRaisesRegex(ValueError,'Angles deviate'):
            examine_record(record,SimpleNamespace(model=m),chart,chart)
