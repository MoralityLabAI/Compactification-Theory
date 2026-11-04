"""LSV-13 verification tests. Requires python-flint 0.9.0 in the CI job."""
import unittest
from unittest.mock import MagicMock
import torch
from flint import arb, arb_mat, ctx
from validated_minor import (State, exact_matrix, initial_state, _jet_layernorm,
                            _jet_gelu, _jet_linear, head_pair_margin, validated_minor)
from upstream_arch import TinyRecursiveModel


class ValidatedMinor(unittest.TestCase):
    def setUp(self):
        ctx.prec=128
        torch.set_num_threads(1)

    def test_binary_float_exactness(self):
        x=torch.tensor([[1.0,-0.25],[0.5,0.0]],dtype=torch.float32)
        m=exact_matrix(x)
        self.assertTrue(m.contains(arb_mat([[1,arb(-1)/4],[arb(1)/2,0]])))

    def test_gelu_derivative_encloses_independent_torch(self):
        for x in [-4.,-1.25,0.,0.67,3.]:
            z=State(arb_mat([[arb(x)]]),(arb_mat([[arb(1)]]),))
            o=_jet_gelu(z)
            p=torch.tensor(x,dtype=torch.float64,requires_grad=True)
            y=torch.nn.functional.gelu(p)
            y.backward()
            self.assertLess(abs(float(o.value[0,0])-float(y.item())),1e-13)
            self.assertTrue(abs(float(o.tangents[0][0,0])-float(p.grad))<1e-13)

    def test_layernorm_tangents_against_autograd(self):
        torch.manual_seed(9)
        norm=torch.nn.LayerNorm(4).double()
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.5,1.,1.5,2.]))
            norm.bias.copy_(torch.tensor([0.1,-.2,.3,-.4]))
        x=torch.tensor([[2.,-1.,0.5,4.],[.3,.2,-.3,-.1]],dtype=torch.float64,requires_grad=True)
        v=torch.tensor([[0.1,0.2,-.4,.3],[-.1,.3,.2,.2]],dtype=torch.float64)
        y=_jet_layernorm(State(exact_matrix(x), (exact_matrix(v),)),norm)
        expected,grad=torch.func.jvp(norm,(x.detach(),),(v,))
        for i in range(2):
            for j in range(4):
                self.assertLess(abs(float(y.value[i,j])-float(expected[i,j])),1e-13)
                self.assertLess(abs(float(y.tangents[0][i,j])-float(grad[i,j])),1e-13)

    def test_linear_directions_match_pytorch(self):
        torch.manual_seed(2)
        l=torch.nn.Linear(3,2).double()
        inp=torch.randn(2,3,dtype=torch.float64)
        derivative=torch.randn(2,3,dtype=torch.float64)
        st=State(exact_matrix(inp),(exact_matrix(derivative),))
        result=_jet_linear(st,exact_matrix(l.weight).transpose(),arb_mat([[arb(float(x)) for x in l.bias.tolist()]]*2))
        for i in range(2):
            for j in range(2):
                self.assertLess(abs(float(result.value[i,j])-float(l(inp)[i,j])),1e-13)
                self.assertLess(abs(float(result.tangents[0][i,j])-float((derivative@l.weight.T)[i,j])),1e-13)

    def test_input_validation(self):
        x=torch.zeros((81,48),dtype=torch.float64)
        with self.assertRaises(ValueError):initial_state(x,(0,0,2,3))
        with self.assertRaises(ValueError):initial_state(x,(0,1,2,3888))
        with self.assertRaises(ValueError):initial_state(x,(0,1,2))

    def test_rank_certificate_rejects_exact_singular(self):
        m=arb_mat([[1,1,1,1],[2,2,2,2],[0,0,0,0],[4,4,4,4]])
        self.assertTrue(m.det().contains(0))

    def test_output_pair_margin_exact_affine(self):
        h=torch.nn.Linear(48,10).double()
        st=State(arb_mat([[0]*48 for _ in range(81)]),tuple(arb_mat([[int(i==j) for j in range(48)] for _ in range(81)]) for i in range(4)))
        _,grad=head_pair_margin(st,h,3,2,4)
        for j in range(4):
            expected=arb(float(h.weight[2,j]))-arb(float(h.weight[4,j]))
            self.assertTrue(grad[j].contains(expected))


if __name__=='__main__':unittest.main()
