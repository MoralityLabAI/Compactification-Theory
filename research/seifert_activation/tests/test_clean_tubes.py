"""SCM-08: independent rational input and source-level clean-state tubes."""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exact_input import (atan_interval,pi_interval,trig_rational,
                         exact_layernorm_point,rmp_input_enclosure,encode_anchors)
from clean_tubes import (clean_state_tubes,verified_local_pair_bounds,_block_constants)
from rmp_lipschitz import layernorm_bounds
from test_rmp_lipschitz import MiniExplicitBlock


class ToyRMP(nn.Module):
    def __init__(self,*,tied=True,positions=True,beta=.1,visits=3,alpha=1.):
        super().__init__()
        self.hidden_size=4
        self.embedding=nn.Embedding(12,4)
        self.input_norm=nn.LayerNorm(4)
        self.use_sinusoidal_positions=positions
        self.blocks=nn.ModuleList([MiniExplicitBlock(d=4,h=2,beta=beta,alpha=alpha)
                for _ in range(1 if tied else visits)])
        self.module_indices=tuple(0 if tied else k for k in range(visits))

    def embed(self,tokens):
        out=self.input_norm(self.embedding(tokens))
        if self.use_sinusoidal_positions:
            d=self.hidden_size
            even=torch.arange(0,d,2,dtype=torch.float32)
            f=torch.exp(-math.log(10000.)*even/d)
            pos=torch.arange(tokens.shape[1],dtype=torch.float32).unsqueeze(1)
            enc=torch.zeros(tokens.shape[1],d)
            enc[:,0::2]=torch.sin(pos*f)
            enc[:,1::2]=torch.cos(pos*f[:enc[:,1::2].shape[1]])
            out=out+enc.unsqueeze(0)
        return out

    def run_clean(self,tokens,reinjection=False):
        enter=self.embed(tokens)
        x=enter; pre=[];mid=[];post=[]
        for idx in self.module_indices:
            block=self.blocks[idx]
            pre.append(x[0].detach().clone())
            found=[]
            hook=block.norm2.register_forward_pre_hook(lambda _mod,args:found.append(args[0][0].detach().clone()))
            try:
                x=block(x)
            finally:
                hook.remove()
            mid.append(found[0])
            if reinjection:
                x=x+enter
            post.append(x[0].detach().clone())
        return pre,mid,post


class RationalInputTests(unittest.TestCase):
    def test_pi_interval(self):
        a,b=pi_interval(20)
        self.assertGreater(a,Fraction("3.14159265358979323846"))
        self.assertLess(b,Fraction("3.14159265358979323847"))
        self.assertLess(b-a,Fraction(1,10**20))

    def test_atan_remainder(self):
        for n in (1,2,6,25):
            a,b=atan_interval(5,n)
            self.assertLessEqual(float(a)-1e-15,math.atan(.2))
            self.assertGreaterEqual(float(b)+1e-15,math.atan(.2))

    def test_trig_rational_many_angles(self):
        for angle in (Fraction(0),Fraction(1,7),Fraction(-2,3),Fraction(3,2),Fraction(9),Fraction(-49,2)):
            for name,fn in [('sin',math.sin),('cos',math.cos)]:
                center,error=trig_rational(angle,name)
                self.assertLessEqual(abs(float(center)-fn(float(angle))),float(error)+2e-14)
                self.assertLess(error,Fraction(1,10**15))

    def test_pi_bound_not_assumed_decimal(self):
        a,b=pi_interval()
        self.assertLess(a,Fraction(22,7))
        self.assertGreater(a,Fraction(223,71))
        self.assertLess(b-a,Fraction(1,10**25))

    def test_layernorm_zero_center(self):
        norm=nn.LayerNorm(4)
        raw=[Fraction(1),Fraction(1),Fraction(1),Fraction(1)]
        c,e=exact_layernorm_point(norm,raw)
        self.assertEqual(c,(Fraction(0),)*4)
        self.assertEqual(e,(Fraction(0),)*4)

    def test_layernorm_exact_rational_enclosure(self):
        norm=nn.LayerNorm(4)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.,2.,1.,.5]))
            norm.bias.copy_(torch.tensor([.25,0.,-.125,.5]))
        raw=[Fraction(-2),Fraction(3),Fraction(0),Fraction(1)]
        c,e=exact_layernorm_point(norm,raw)
        actual=norm(torch.tensor([float(v) for v in raw],dtype=torch.float32))
        for i in range(4):
            self.assertLessEqual(abs(float(c[i])-float(actual[i].detach())),float(e[i])+2e-6)
        self.assertLess(max(e),Fraction(1,10**10))

    def test_encode_rounding_is_explicit(self):
        tiny=Fraction(1,10**30)
        tensor,errors=encode_anchors([(tiny,1+tiny)],[(0,0)])
        self.assertGreater(errors[0],0)
        self.assertLessEqual(abs(Fraction(*float(tensor[0,0]).as_integer_ratio())-tiny),errors[0])

    def test_input_anchor_with_positional_table(self):
        torch.manual_seed(11)
        model=ToyRMP().eval()
        tokens=torch.tensor([1,3,5,6,7])
        source=rmp_input_enclosure(model,tokens)
        actual=model.embed(tokens[None])[0].double()
        self.assertEqual(source.scope,'ideal_real_with_exact_dyadic_frequency_table')
        self.assertEqual(source.center.shape,(5,4))
        for i in range(5):
            self.assertLessEqual(float((source.center[i]-actual[i]).norm().detach()),float(source.radii[i])+5e-6)
        self.assertLess(max(source.radii),Fraction(1,10**9))

    def test_nonreset_carry_refused(self):
        model=ToyRMP().eval()
        model.carry_policy="persistent"
        with self.assertRaisesRegex(ValueError,"reset carry"):
            rmp_input_enclosure(model,torch.tensor([1,2]))

    def test_wrong_frequency_refused(self):
        model=ToyRMP().eval()
        with self.assertRaises(ValueError):
            rmp_input_enclosure(model,torch.tensor([1,2]),positional_frequency_table=torch.ones(4))

    def test_input_anchor_no_positions(self):
        model=ToyRMP(positions=False).eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3]))
        self.assertEqual(source.center.shape,(3,4))
        self.assertEqual(source.frequency_sha256,__import__('hashlib').sha256(b'').hexdigest())


class CertifiedCleanTubes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(91)
        cls.model=ToyRMP().eval()
        cls.tokens=torch.tensor([1,2,3,4,5])
        cls.source=rmp_input_enclosure(cls.model,cls.tokens)

    def _assert_encloses(self,model,tokens,reinjection=False):
        source=rmp_input_enclosure(model,tokens)
        tubes=clean_state_tubes(model,source,fixed_input_reinjection=reinjection)
        clean=model.run_clean(tokens[None],reinjection=reinjection)
        for site,(centers,errors) in enumerate(((tubes.pre_centers,tubes.pre_center_errors),
                                                (tubes.mid_centers,tubes.mid_center_errors))):
            for visit in range(len(centers)):
                for pos in range(len(tokens)):
                    observed=clean[site][visit][pos].double()
                    claimed=centers[visit][pos].double()
                    self.assertLessEqual(float((observed-claimed).norm()),float(errors[visit][pos])+2e-5)
        return tubes

    def test_clean_tubes_tied(self):
        self._assert_encloses(self.model,self.tokens)

    def test_clean_tubes_untied_reinjection(self):
        torch.manual_seed(42)
        model=ToyRMP(tied=False,beta=.15).eval()
        tubes=self._assert_encloses(model,torch.tensor([3,4,5,6,7]),reinjection=True)
        self.assertGreater(max(tubes.post_radii[2]),max(tubes.post_radii[0]))

    def test_attention_convexity_bound(self):
        block=self.model.blocks[0]
        a,b,A,H=_block_constants(block,5)
        y=torch.randn(1,5,4)
        y=torch.nn.functional.layer_norm(y,(4,),weight=block.norm1.weight,bias=block.norm1.bias)
        QKV=block.qkv(y).reshape(1,5,3,2,2)
        Q,K,V=[t.transpose(1,2) for t in QKV.unbind(2)]
        scores=Q@K.transpose(-2,-1)/math.sqrt(2)
        scores=scores.masked_fill(torch.triu(torch.ones(5,5,dtype=torch.bool),1),float('-inf'))
        att=block.attention_out((torch.softmax(scores,-1)@V).transpose(1,2).reshape(1,5,4))
        self.assertLessEqual(float(att.norm(dim=-1).max().detach()),float(A)+1e-6)

    def test_beta_zero_exact_clean(self):
        model=ToyRMP(beta=0).eval()
        source=rmp_input_enclosure(model,torch.tensor([1,2,3]))
        tubes=clean_state_tubes(model,source)
        self.assertEqual(tubes.post_radii[0],source.radii)

    def test_unknown_input_scope_refused(self):
        from dataclasses import replace
        wrong=replace(self.source,scope='unverified_capture')
        with self.assertRaises(ValueError):
            clean_state_tubes(self.model,wrong)

    def test_verified_scoped_bounds(self):
        chart=torch.eye(4,dtype=torch.float32)[:,:2]
        answer=verified_local_pair_bounds(self.model,self.source,chart=chart,axes=chart,
             amplitude='1/1000',observed_tokens=(1,2,4),injected_token=0)
        self.assertEqual(answer['schema'],'scm08_derived_ideal_clean_enclosures_v1')
        self.assertTrue(answer['center_enclosures_verified_by_this_program'])
        self.assertFalse(answer['floating_execution_certified'])
        self.assertEqual(len(answer['clean_enclosures']),3)

    def test_causal_negative_control(self):
        chart=torch.eye(4)[:,:2]
        answer=verified_local_pair_bounds(self.model,self.source,chart=chart,axes=chart,
             amplitude='1',observed_tokens=(0,1,2),injected_token=4)
        self.assertTrue(all(all(x=='0' for x in r.values()) for r in answer['per_visit_pair_bounds']))

    def test_no_unvalidated_attention_bias(self):
        block=MiniExplicitBlock()
        block.mlp[0]=nn.Linear(4,16,bias=True)
        with self.assertRaises(ValueError):
            _block_constants(block,5)


if __name__=='__main__':
    unittest.main()
