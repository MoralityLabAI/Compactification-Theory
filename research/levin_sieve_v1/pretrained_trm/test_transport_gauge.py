"""LSV-16 mathematical invariant tests; numerical tests run in CI with Arb."""
import unittest
from fractions import Fraction as Q
import torch
from flint import arb,arb_mat,ctx
from transport_gauge import (centered_abs_bounds, as_dyadic, upper_decimal,
    layernorm_difference_bounds, CertifiedTransport, Enclosure, goal_cell_radius)
from upstream_arch import TinyRecursiveModel

class Tests(unittest.TestCase):
    def setUp(self):
        ctx.prec=160

    def test_exact_dyadic(self):
        self.assertEqual(as_dyadic(arb(1)/8),Q(1,8))
        self.assertGreaterEqual(upper_decimal(arb('0.0000000000001')),Q(1,10**13))

    def test_centered_bound_extremal_signs(self):
        for arr in ((Q(1),Q(3),Q(2)),(Q(0),Q(1),Q(0)),(Q(3),Q(3),Q(3))):
            c,r=centered_abs_bounds([arb(str(x)) for x in arr])
            for mask in range(8):
                e=[arr[i] * (-1 if mask>>i&1 else 1) for i in range(3)]
                avg=sum(e)/3
                for j in range(3):
                    self.assertGreaterEqual(as_dyadic(c[j]),abs(e[j]-avg))

    def test_uniform_noise_exact_shift(self):
        norm=torch.nn.LayerNorm(3,dtype=torch.float64)
        x=arb_mat([[arb(1),arb(2),arb(3)]])
        d=arb_mat([[arb(0),arb(0),arb(0)]])
        b=layernorm_difference_bounds(x,d,norm)
        self.assertTrue(all(as_dyadic(z)==0 for z in b.tolist()[0]))

    def test_layernorm_bound_against_exhaustive_vertices(self):
        norm=torch.nn.LayerNorm(3,dtype=torch.float64)
        with torch.no_grad():
            norm.weight[:]=torch.tensor([1.3,-0.7,0.9],dtype=torch.float64)
            norm.bias[:]=torch.tensor([0.2,-0.4,0.1],dtype=torch.float64)
        x=[Q(1),Q(-1,2),Q(3,4)]
        e=[Q(1,50),Q(1,100),Q(1,40)]
        ub=layernorm_difference_bounds(arb_mat([[arb(str(z)) for z in x]]),
                    arb_mat([[arb(str(z)) for z in e]]),norm).tolist()[0]
        center=norm(torch.tensor([[float(a) for a in x]],dtype=torch.float64))[0]
        for mask in range(8):
            delta=[float(e[j])*(-1 if (mask>>j)&1 else 1) for j in range(3)]
            trial=norm(torch.tensor([[float(x[j])+delta[j] for j in range(3)]],dtype=torch.float64))[0]
            for j in range(3):
                self.assertLess(float(abs(center[j]-trial[j])),float(ub[j])+1e-12)

    def test_transport_zero_radius_and_shapes(self):
        model=TinyRecursiveModel().double().eval()
        net=CertifiedTransport(model)
        torch.manual_seed(17)
        x=torch.randn(81,48,dtype=torch.float64)/2
        # We don't call from_stage_one, which requires positive radius; zero-radius
        # transport is an exact affine expression for a known clean state.
        box=Enclosure(arb_mat([[arb(float(z)) for z in row] for row in x.tolist()]),
            arb_mat(81,48))
        out=net.pre_second_ln(box)
        self.assertEqual(out.rad.nrows(),81)
        self.assertEqual(out.rad.ncols(),48)
        self.assertEqual(goal_cell_radius(out,0),0)

    def test_input_rejected(self):
        model=TinyRecursiveModel().double().eval()
        net=CertifiedTransport(model)
        with self.assertRaises(ValueError):
            net.from_stage_one(torch.zeros(81,48,dtype=torch.float64),'0')
        with self.assertRaises(ValueError):
            net.from_stage_one(torch.zeros(80,48,dtype=torch.float64),'1e-8')

if __name__=='__main__':unittest.main()
