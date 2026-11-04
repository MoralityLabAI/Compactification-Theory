import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
import torch
from flint import arb,ctx
from arb_engine import TaylorNet,endpoint,verified_spectral,aq,evaluate_case
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pretrained_trm'))
from upstream_arch import TinyRecursiveModel


class TestArbTaylor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=128;torch.set_num_threads(1);torch.manual_seed(471)
        cls.model=TinyRecursiveModel().double().eval()
        cls.net=TaylorNet(cls.model)
        cls.stage=torch.randn(81,48,dtype=torch.float64)
    def test_verified_spectral_contains_float_diagnostic(self):
        for m in [torch.eye(3),torch.tensor([[1.,2.],[-3.,4.]])]:
            bound,receipt=verified_spectral(m)
            self.assertGreaterEqual(float(bound),float(torch.linalg.matrix_norm(m.double(),ord=2)))
            self.assertGreater(Q(receipt['ldl_pivot_lower_min']),0)
    def test_reverse_matches_torch(self):
        h=self.stage.clone().requires_grad_(True)
        z=self.model.block(h.unsqueeze(0))[0]
        margin=self.model.head(z)[0,3]-self.model.head(z)[0,7]
        g=torch.autograd.grad(margin,h)[0]
        final,tape=self.net.forward(self.stage,steps=1)
        adj=self.net.margin_adjoint(tape,0,3,7)['gradient']
        err=max(abs(float(adj[i,j])-float(g[i,j])) for i in range(81) for j in range(48))
        self.assertLess(err,1e-12)
    def test_taylor_remainder_covers_points(self):
        final,tape=self.net.forward(self.stage,steps=1)
        adj=self.net.margin_adjoint(tape,0,3,7)
        clean=self.net.logits(final,0)
        g=adj['gradient']
        r=Q(1,100000)
        local=self.net.transport(tape,r,3888)
        remainder=sum((q*t['defect'] for q,t in zip(adj['q'],local)),arb(0)).upper()
        for seed in range(3):
            gen=torch.Generator().manual_seed(seed)
            damage=(torch.rand((81,48),generator=gen,dtype=torch.float64)*2-1)*float(r)
            damaged=self.stage+damage
            out,_=self.net.forward(damaged,steps=1)
            val=self.net.logits(out,0)
            linear=sum((g[i,j]*(arb(float(damaged[i,j]))-arb(float(self.stage[i,j])))
                        for i in range(81) for j in range(48)),arb(0))
            err=val[3]-val[7]-(clean[3]-clean[7])-linear
            self.assertLess(float(abs(err).upper()),float(remainder))
    def test_scope_and_all_rivals(self):
        result=evaluate_case(self.net,self.stage,0,[Q(1,10**14)],steps=1)
        self.assertEqual(len(result['pairs']),9)
        self.assertEqual(len(result['regions']),2)
        self.assertTrue(all(x['certified'] for x in result['regions']))

if __name__=='__main__':unittest.main()
