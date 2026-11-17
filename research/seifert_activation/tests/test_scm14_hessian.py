"""SCM-14 independent analytic and autograd checks on frozen synthetic decoders."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from scm14_hessian import (
    ScalarJet, layernorm_jet_bounds, softmax_jet_bound,
    head_jet_bound, hessian_disk_bounds,
)
from scm09_reference import DeterministicRMP
from exact_input import rmp_input_enclosure
from ball_centered import verified_ball_tubes
from validated_forward import verified_injection_amplitude


def source_fixture(*, d=8, seq=8, visits=3, tied=True, reinjection=False,
                   beta=.1):
    model = DeterministicRMP(d=d,seq=seq,visits=visits,tied=tied,
                             reinjection=reinjection,beta=beta)
    freqs=torch.tensor([2.0**(-k) for k in range((d+1)//2)],dtype=torch.float32)
    source=rmp_input_enclosure(model,model.tokens,positional_frequency_table=freqs)
    tubes=verified_ball_tubes(model,source,fixed_input_reinjection=reinjection,
                              max_seq=seq,max_dim=d,max_visits=visits)
    return model,source,tubes


def apply_block(block, x):
    n,d=x.shape
    nh=block.num_heads
    head=d//nh
    normalized=block.norm1(x)
    qkv=block.qkv(normalized).reshape(n,3,nh,head)
    q,k,v=[qkv[:,idx].transpose(0,1) for idx in range(3)]
    logits=q@k.transpose(-2,-1)/math.sqrt(head)
    causal=torch.triu(torch.ones(n,n,dtype=torch.bool,device=x.device),diagonal=1)
    attention=torch.softmax(logits.masked_fill(causal,float('-inf')),dim=-1)
    heads=attention@v
    attended=block.attention_out(heads.transpose(0,1).reshape(n,d))
    mid=block.alpha*x+block.beta*attended
    return block.alpha*mid+block.beta*block.mlp(block.norm2(mid))


def projected_source_forward(model,source,u,amp,axes,chart,*,inject_token=1,
                             inject_after_visit=0):
    current=source.center.clone().double()
    initial=current.clone()
    readouts=[]
    for visit,idx in enumerate(model.module_indices):
        current=apply_block(model.blocks[idx],current)
        if model.reinjection:
            current=current+initial
        if visit==inject_after_visit:
            patch=(axes.double()@u)*float(amp)
            current=current.clone()
            current[inject_token]=current[inject_token]+patch
        readouts.append(current@chart.double())
    return readouts


class SCM14AnalyticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_softmax_first_second_derivative_inequalities(self):
        for j,h in [(0,0),('1/2','3/4'),(1,2),(2,5)]:
            got=softmax_jet_bound(ScalarJet(Fraction(j),Fraction(h)))
            self.assertEqual(got.j,Fraction(j))
            self.assertEqual(got.h,Fraction(h)+2*Fraction(j)**2)

    def test_softmax_second_derivative_by_independent_autograd(self):
        s=torch.tensor([.1,-.3,.2,-.8],dtype=torch.float64,requires_grad=True)
        for a,b in [(torch.tensor([1.,-1.,.5,-.25],dtype=torch.float64),
                     torch.tensor([-.25,.5,-.7,1.],dtype=torch.float64))]:
            for k in range(4):
                hess=torch.autograd.functional.hessian(lambda v:torch.softmax(v,dim=-1)[k],s)
                # Test the l1 bound for the whole probability vector.
            hv=torch.stack([torch.autograd.functional.hessian(
                    lambda v:torch.softmax(v,dim=-1)[k],s) for k in range(4)])
            got=(hv@b).matmul(a).abs().sum().item()
            bound=2*float(a.abs().max())*float(b.abs().max())
            self.assertLessEqual(got,bound+1e-12)

    def test_layernorm_bound_covers_high_precision_derivatives(self):
        import mpmath as mp
        mp.mp.dps=70
        ln=torch.nn.LayerNorm(4).double().eval()
        with torch.no_grad():
            ln.weight.copy_(torch.tensor([1.1,.9,1.3,.8],dtype=torch.float64))
            ln.bias.copy_(torch.tensor([.1,-.3,.2,0.],dtype=torch.float64))
        center=torch.tensor([.1,-.5,.7,1.9],dtype=torch.float64)
        r=Fraction(1,10)
        l1,l2=layernorm_jet_bounds(ln,center,r)
        self.assertGreater(l1,0)
        self.assertGreater(l2,0)
        # Independently differentiate a 4D mpmath LN along two directions.
        c=[mp.mpf(str(float(x))) for x in center]
        A=[mp.mpf(x) for x in ['.3','-.1','.4','.1']]
        B=[mp.mpf(x) for x in ['-.2','.5','.1','-.2']]
        na=mp.sqrt(sum(x*x for x in A));nb=mp.sqrt(sum(x*x for x in B))
        A=[x/na for x in A];B=[x/nb for x in B]
        gam=[mp.mpf(str(float(x))) for x in ln.weight.detach()]
        eps=mp.mpf(str(float(ln.eps)))
        def f(i,t,s):
            x=[c[k]+t*A[k]+s*B[k] for k in range(4)]
            avg=sum(x)/4
            variance=sum((v-avg)**2 for v in x)/4
            return gam[i]*(x[i]-avg)/mp.sqrt(variance+eps)
        for i in range(4):
            d2=mp.diff(lambda t,s:f(i,t,s),(0,0),(1,1))
            self.assertLessEqual(abs(d2),mp.mpf(str(float(l2)))+mp.mpf('1e-35'))

    def test_layernorm_global_fallback_at_zero_center(self):
        ln=torch.nn.LayerNorm(8).double().eval()
        got=layernorm_jet_bounds(ln,torch.zeros(8,dtype=torch.float64),Fraction(1))
        self.assertGreaterEqual(got[1],0)
        self.assertLess(got[1],Fraction(500000))

    def test_uniform_head_bound_rejects_negative_norm(self):
        with self.assertRaises(ValueError):
            head_jet_bound(qnorm=-1,knorm=1,vnorm=1,normalized_max=1,
                    query=ScalarJet(Fraction(0),Fraction(0)),
                    keys=ScalarJet(Fraction(0),Fraction(0)),
                    values=ScalarJet(Fraction(0),Fraction(0)),head_dim=4)

    def test_causal_prefix_zero_and_finite_remainder_tied(self):
        model,source,tubes=source_fixture()
        axes=torch.eye(8,dtype=torch.float32)[:,:2]
        amp=verified_injection_amplitude(tubes,injected_token=1,after_visit=0,
                                           radius_fraction=Fraction(1,10**5))
        cert=hessian_disk_bounds(model,tubes,axes=axes,chart=axes,amplitude=amp,
                         injected_token=1,observed_tokens=(0,2,4))
        self.assertEqual(cert['schema'],'scm14/two_coordinate_source_hessian/v1')
        self.assertTrue(all(r['causal_prefix_zero'] for r in cert['per_visit']))
        self.assertEqual(Fraction(cert['per_visit'][0]['observed']['2']['second_derivative_upper']),0)
        self.assertGreater(Fraction(cert['per_visit'][-1]['observed']['2']['second_derivative_upper']),0)
        self.assertFalse(cert['float32_cuda_execution_certified'])

    def test_source_hessian_covers_independent_autograd_tied(self):
        for tied,reinjection in [(True,False),(False,True)]:
            model,source,tubes=source_fixture(visits=3,tied=tied,reinjection=reinjection)
            axes=torch.eye(8,dtype=torch.float32)[:,:2]
            amp=verified_injection_amplitude(tubes,injected_token=1,after_visit=0,
                                                radius_fraction=Fraction(1,10**4))
            cert=hessian_disk_bounds(model,tubes,axes=axes,chart=axes,
                                     amplitude=amp,injected_token=1,
                                     observed_tokens=(0,2,4))
            model.double()  # Independent float64 forward after source-bound derivation.
            for u0 in ((0.0,0.0),(.3,.4),(-.5,.25)):
                u=torch.tensor(u0,dtype=torch.float64)
                for token in (0,2,4):
                    hess=[]
                    for dim in (0,1):
                        def scalar(z):
                            return projected_source_forward(model,source,z,amp,axes,axes)[-1][token,dim]
                        hess.append(torch.autograd.functional.hessian(scalar,u))
                    for av,bv in (([1.,0.],[0.,1.]),([.6,.8],[.8,-.6])):
                        a=torch.tensor(av,dtype=torch.float64);b=torch.tensor(bv,dtype=torch.float64)
                        actual=torch.linalg.vector_norm(torch.stack([a@H@b for H in hess])).item()
                        upper=float(Fraction(cert['per_visit'][-1]['observed'][str(token)]['second_derivative_upper']))
                        self.assertLessEqual(actual,upper+1e-10,
                            f'Source Hessian bound fails for tied={tied},token={token},u={u0}')

    def test_zero_amplitude_zero_hessian(self):
        model,source,tubes=source_fixture(visits=3)
        axes=torch.eye(8)[:,:2]
        cert=hessian_disk_bounds(model,tubes,axes=axes,chart=axes,
                                 amplitude=0,injected_token=1,observed_tokens=(2,4))
        self.assertTrue(all(Fraction(q['second_derivative_upper'])==0
                         for r in cert['per_visit'] for q in r['observed'].values()))

    def test_bad_source_tubes_refused(self):
        from dataclasses import replace
        model,source,tubes=source_fixture(visits=2)
        axes=torch.eye(8)[:,:2]
        damaged=replace(tubes, input_scope='unvalidated_capture')
        with self.assertRaisesRegex(ValueError,'Source-validated'):
            hessian_disk_bounds(model,damaged,axes=axes,chart=axes,
                       amplitude='1/100000',injected_token=1,observed_tokens=(2,4))

    def test_nonfinite_axes_refused(self):
        model,source,tubes=source_fixture(visits=2)
        axes=torch.eye(8)[:,:2]
        axes[0,0]=float('nan')
        with self.assertRaises(ValueError):
            hessian_disk_bounds(model,tubes,axes=axes,chart=axes,
                      amplitude='1/100000',injected_token=1,observed_tokens=(2,4))


if __name__=='__main__':unittest.main()
