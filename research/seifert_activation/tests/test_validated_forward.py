"""SCM-09 independent numerical references and exact interval stress checks."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import random
import sys
import unittest

import mpmath as mp
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_clean_tubes import ToyRMP
from exact_input import rmp_input_enclosure
from clean_tubes import clean_state_tubes
from validated_forward import (
    I, _exp_point, phi_point, gelu_interval, verified_centered_tubes,
    verified_centered_pair_bounds, verified_injection_amplitude,
    centered_null_disk, centered_pair_lower,
)


def mfrac(x):
    if not isinstance(x,Fraction):
        x=Fraction(x)
    return mp.mpf(x.numerator)/x.denominator


def mf(x):
    return mfrac(Fraction(*float(x.detach() if isinstance(x,torch.Tensor) else x).as_integer_ratio()))


def m_linear(W,x):
    return [sum((mf(w)*a for w,a in zip(row,x)),mp.mpf(0))
            for row in W.detach().cpu().tolist()]


def m_ln(norm,x):
    d=len(x)
    mean=sum(x,mp.mpf(0))/d
    var=sum(((v-mean)**2 for v in x),mp.mpf(0))/d
    scale=mp.sqrt(var+mf(norm.eps))
    return [(val-mean)*mf(g)/scale+mf(b) for val,g,b in
            zip(x,norm.weight.detach().cpu().tolist(),norm.bias.detach().cpu().tolist())]


def mp_clean_reference(model,tokens,*,reinjection=False):
    """Independent 90-digit reference; no interval operations used in evaluator."""
    d=model.hidden_size
    if model.use_sinusoidal_positions:
        even=torch.arange(0,d,2,dtype=torch.float32)
        freqs=torch.exp(-math.log(10000.)*even/d).tolist()
    else:
        freqs=[]
    inp=[]
    for pos,idx in enumerate(tokens.tolist()):
        x=m_ln(model.input_norm, [mf(w) for w in model.embedding.weight[int(idx)]])
        if freqs:
            x=[v+(mp.sin(pos*mf(freqs[k//2])) if k%2==0 else mp.cos(pos*mf(freqs[k//2])))
               for k,v in enumerate(x)]
        inp.append(x)
    x=inp
    pre=[];middle=[];post=[]
    for idx in model.module_indices:
        b=model.blocks[idx]
        pre.append(x)
        n1=[m_ln(b.norm1,row) for row in x]
        qkv=[m_linear(b.qkv.weight,row) for row in n1]
        h=b.num_heads;dh=b.head_size
        att=[]
        for i in range(len(x)):
            vals=[]
            for head in range(h):
                q=qkv[i][head*dh:(head+1)*dh]
                score=[]
                for j in range(i+1):
                    k=qkv[j][d+head*dh:d+(head+1)*dh]
                    score.append(sum((a*c for a,c in zip(q,k)),mp.mpf(0))/mp.sqrt(dh))
                weight=[mp.exp(s-max(score)) for s in score]
                total=sum(weight,mp.mpf(0))
                vals.extend(sum((weight[j]*qkv[j][2*d+head*dh+coord] for j in range(i+1)),
                                mp.mpf(0))/total for coord in range(dh))
            att.append(m_linear(b.attention_out.weight,vals))
        a=mf(b.alpha);beta=mf(b.beta)
        mid=[[a*v+beta*av for v,av in zip(xrow,arow)] for xrow,arow in zip(x,att)]
        middle.append(mid)
        n2=[m_ln(b.norm2,row) for row in mid]
        hidden=[[z*(1+mp.erf(z/mp.sqrt(2)))/2 for z in m_linear(b.mlp[0].weight,row)]
                for row in n2]
        mlp=[m_linear(b.mlp[2].weight,row) for row in hidden]
        x=[[a*v+beta*m for v,m in zip(row,update)] for row,update in zip(mid,mlp)]
        if reinjection:
            x=[[v+iv for v,iv in zip(row,initial)] for row,initial in zip(x,inp)]
        post.append(x)
    return pre,middle,post


class IntervalPrimitiveTests(unittest.TestCase):
    def test_integer_signed_rounding_and_negative_multiply(self):
        for x in (Fraction(-1,7),Fraction(1,11),Fraction(-50,3),Fraction(0)):
            b=I.of(x)
            self.assertLessEqual(b.lo,x)
            self.assertGreaterEqual(b.hi,x)
        a=I.of('-0.2','0.7');b=I.of('-1.4','0.3')
        m=a*b
        for x in (a.lo,a.hi):
            for y in (b.lo,b.hi):
                self.assertLessEqual(m.lo,x*y)
                self.assertGreaterEqual(m.hi,x*y)

    def test_division_and_sqrt(self):
        x=I.of('2.1','2.2')
        y=(x.square()+I.of('1/3')).sqrt()
        for z in (Fraction('2.1'),Fraction('2.2')):
            true=mp.sqrt(mfrac(z*z+Fraction(1,3)))
            self.assertLessEqual(mfrac(y.lo),true)
            self.assertGreaterEqual(mfrac(y.hi),true)
        with self.assertRaisesRegex(ValueError,'crosses zero'):
            I.of(-1,1).reciprocal()

    def test_exp_bounds_over_signs(self):
        with mp.workdps(90):
            for arg in ('-12','-3','-1/7','0','1/4','2','6','12'):
                x=Fraction(arg);r=_exp_point(x);actual=mp.exp(mfrac(x))
                self.assertLessEqual(mfrac(r.lo),actual)
                self.assertGreaterEqual(mfrac(r.hi),actual)
                self.assertLess(float(r.hi-r.lo),max(1e-13,1e-12*float(actual)))

    def test_phi_bounds_and_gelu(self):
        with mp.workdps(90):
            for arg in ('-6','-2.3','-1/10','0','1/5','1.5','5.9','7'):
                x=Fraction(arg);p=phi_point(x)
                actual=(1+mp.erf(mfrac(x)/mp.sqrt(2)))/2
                self.assertLessEqual(mfrac(p.lo),actual)
                self.assertGreaterEqual(mfrac(p.hi),actual)
                gelu=gelu_interval(I.of(x))
                actual_g=mfrac(x)*actual
                self.assertLessEqual(mfrac(gelu.lo),actual_g)
                self.assertGreaterEqual(mfrac(gelu.hi),actual_g)

    def test_gelu_interval_random(self):
        random.seed(63)
        with mp.workdps(70):
            for _ in range(40):
                a=Fraction(random.randint(-300,200),100)
                b=a+Fraction(random.randint(1,25),100)
                result=gelu_interval(I.of(a,b))
                for k in range(9):
                    x=a+(b-a)*k/8
                    true=mfrac(x)*(1+mp.erf(mfrac(x)/mp.sqrt(2)))/2
                    self.assertLessEqual(mfrac(result.lo),true)
                    self.assertGreaterEqual(mfrac(result.hi),true)

    def test_reject_domain_overflow(self):
        with self.assertRaises(ValueError):
            _exp_point(Fraction(40))


class VerifiedModelTests(unittest.TestCase):
    def run_mp_case(self,seed,*,tied=True,visits=3,beta=.1,reinjection=False,positions=True):
        with mp.workdps(90):
            torch.manual_seed(seed)
            model=ToyRMP(tied=tied,visits=visits,beta=beta,positions=positions).eval()
            tokens=torch.tensor([1,2,3,4,5])
            source=rmp_input_enclosure(model,tokens)
            tube=verified_centered_tubes(model,source,fixed_input_reinjection=reinjection)
            refs=mp_clean_reference(model,tokens,reinjection=reinjection)
            for centers,errs,ref in zip((tube.pre_centers,tube.mid_centers,tube.post_centers),
                                         (tube.pre_center_errors,tube.mid_center_errors,tube.post_center_errors),refs):
                for visit in range(visits):
                    for j in range(len(tokens)):
                        delta=mp.sqrt(sum((ref[visit][j][k]-mf(centers[visit][j,k]))**2
                                          for k in range(model.hidden_size)))
                        self.assertLessEqual(delta,mfrac(errs[visit][j]))
            return model,source,tube,tokens

    def test_exact_real_tied_clean(self):
        self.run_mp_case(12,tied=True)

    def test_exact_real_untied_reinjection(self):
        self.run_mp_case(13,tied=False,reinjection=True)

    def test_exact_real_no_positions(self):
        self.run_mp_case(14,positions=False)

    def test_nonunit_alpha(self):
        torch.manual_seed(18)
        model=ToyRMP(beta=.05,alpha=.7).eval()
        tokens=torch.tensor([1,2,3,4,5]);source=rmp_input_enclosure(model,tokens)
        tube=verified_centered_tubes(model,source)
        with mp.workdps(70):
            ref=mp_clean_reference(model,tokens)
            for visit in range(3):
                dist=mp.sqrt(sum((ref[2][visit][2][k]-mf(tube.post_centers[visit][2,k]))**2
                                 for k in range(4)))
                self.assertLessEqual(dist,mfrac(tube.post_center_errors[visit][2]))

    def test_beta_zero_nearly_exact(self):
        torch.manual_seed(41)
        model=ToyRMP(beta=0).eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        tube=verified_centered_tubes(model,source)
        self.assertLess(float(max(tube.post_center_errors[2])),1e-12)

    def test_centered_width_improvement_over_scm08(self):
        torch.manual_seed(55)
        model=ToyRMP(visits=3,beta=.1).eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        old=clean_state_tubes(model,source)
        new=verified_centered_tubes(model,source)
        self.assertLess(float(max(new.post_center_errors[2])),1e-11)
        self.assertGreater(float(max(old.post_radii[2])),.01)

    def test_source_only_injection_rms_bound(self):
        model,source,tube,tokens=self.run_mp_case(99)
        with mp.workdps(70):
            ref=mp_clean_reference(model,tokens)
            for visit in range(3):
                a=verified_injection_amplitude(tube,injected_token=2,
                              after_visit=visit,radius_fraction=Fraction(1,20))
                rms=mp.sqrt(sum((v*v for v in ref[2][visit][2]),mp.mpf(0))/4)
                self.assertGreaterEqual(mfrac(a),rms/20)

    def test_centered_null_certificate_and_prefix(self):
        model,source,tube,tokens=self.run_mp_case(101)
        chart=torch.eye(4)[:,:2]
        bound=verified_centered_pair_bounds(model,tube,chart=chart,axes=chart,
                        amplitude=Fraction(1,10**7),observed_tokens=(1,2,3),injected_token=0)
        self.assertTrue(bound['center_enclosures_verified_by_this_program'])
        self.assertTrue(all(not r['floating_execution_certified'] for r in
                            centered_null_disk(tube,chart,(1,2,3),bound['per_visit_pair_bounds'])))
        negative=verified_centered_pair_bounds(model,tube,chart=chart,axes=chart,
                        amplitude=1,observed_tokens=(0,1,2),injected_token=4)
        for row in negative['per_visit_pair_bounds']:
            self.assertTrue(all(v=='0' for v in row.values()))
        reports=centered_null_disk(tube,chart,(0,1,2),negative['per_visit_pair_bounds'])
        self.assertTrue(all(r['null_braid_ideal_real'] for r in reports))

    def test_refuses_oversized_proof(self):
        model=ToyRMP().eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        with self.assertRaisesRegex(ValueError,'size cap'):
            verified_centered_tubes(model,source,max_seq=4)

    def test_refuses_changed_mlp(self):
        model=ToyRMP().eval()
        model.blocks[0].mlp[1]=nn.ReLU()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        with self.assertRaisesRegex(ValueError,'GELU'):
            verified_centered_tubes(model,source)

    def test_rejects_unverified_source(self):
        from dataclasses import replace
        model=ToyRMP().eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3,4,5]))
        with self.assertRaisesRegex(ValueError,'certified'):
            verified_centered_tubes(model,replace(source,scope='float_capture_unverified'))

    def test_no_extrapolation_to_float_execution(self):
        model,source,tube,tokens=self.run_mp_case(23)
        self.assertEqual(tube.method,'rational_outward_interval_exact_real')
        assert not hasattr(tube,'floating_execution_certified')


if __name__=='__main__':
    unittest.main()
