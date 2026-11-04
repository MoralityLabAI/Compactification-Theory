import unittest
from fractions import Fraction as Q
import torch
from flint import arb,arb_mat,ctx
from arb_engine import TaylorNet,Mix,LN,up
from projected import curvature_data,goal_curvature,prepare,evaluate,forward_from_arb
from upstream_arch import TinyRecursiveModel


class TestProjectedCurvature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=128;torch.set_num_threads(1);torch.manual_seed(507)
        cls.model=TinyRecursiveModel().double().eval()
        cls.net=TaylorNet(cls.model)
        cls.stage=torch.randn(81,48,dtype=torch.float64)
    def test_projection_removes_layernorm_common_covector(self):
        norm=torch.nn.LayerNorm(48).double()
        _,op=self.net.ln(arb_mat([[arb(i)/48 for i in range(48)]]),norm,'test')
        G=arb_mat([[arb(1) for _ in range(48)]])
        data=curvature_data(self.net,op,G)
        self.assertTrue(data['norms'][0].is_zero())
        self.assertTrue(goal_curvature(op,data,arb('0.01')).is_zero())
    def test_projected_scalar_curvature_covers_directional_hessian(self):
        final,tape=self.net.forward(self.stage,steps=1)
        for idx in (0,1,2,3):
            op=tape[idx]
            if idx==0:X=self.stage
            elif idx==1:X=self.stage+self.model.block.tok_mix(self.stage.T).T
            elif idx==2:X=self.model.block.ln1(self.stage+self.model.block.tok_mix(self.stage.T).T)
            else:
                first=self.model.block.ln1(self.stage+self.model.block.tok_mix(self.stage.T).T)
                X=first+self.model.block.chan_mix(first)
            x=X.detach().clone().requires_grad_(True)
            c=torch.randn_like(x);v=torch.randn_like(x);v=v/torch.linalg.vector_norm(v)
            if idx==0:y=x+self.model.block.tok_mix(x.T).T
            elif idx==1:y=self.model.block.ln1(x)
            elif idx==2:y=x+self.model.block.chan_mix(x)
            else:y=self.model.block.ln2(x)
            grad=torch.autograd.grad((c*y).sum(),x,create_graph=True)[0]
            hv=torch.autograd.grad((grad*v).sum(),x)[0]
            directional=abs(float((hv*v).sum()))
            G=arb_mat(c.detach().tolist())
            data=curvature_data(self.net,op,G)
            bound=goal_curvature(op,data,arb('0.001'))
            self.assertLess(directional,float(bound))
    def test_projection_improves_only_remainder_not_contract(self):
        rec,_=evaluate(self.net,self.stage,0,[Q(1,10**10)],steps=1)
        self.assertEqual(len(rec['pairs']),9)
        for region in rec['regions']:
            self.assertTrue(region['certified'])
            for pair in region['pairs']:
                self.assertLessEqual(Q(pair['remainder_upper']),Q(pair['generic_remainder_upper']))
    def test_exact_arb_forward_path_matches_base(self):
        from validated_minor import exact_matrix
        a,_=self.net.forward(self.stage,steps=1)
        b=forward_from_arb(self.net,exact_matrix(self.stage),steps=1)
        for i in range(81):
            for j in range(48):self.assertTrue((a[i,j]-b[i,j]).contains(0))

if __name__=='__main__':unittest.main()
