"""Arb graph-path checks for the final LayerNorm intervention bus."""
import unittest
import torch
from flint import arb,ctx
from run_gauge import pre_final_ln2,centered_arbs,exact_weights
from upstream_arch import TinyRecursiveModel
from validated_minor import CertifiedBlock
from gauge_quotient import Bound,certificate_for_case


class TestGaugeArb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(20261104)
        cls.model=TinyRecursiveModel().double().eval()
        cls.z1=torch.randn(81,48,dtype=torch.float64)*0.2
        ctx.prec=160

    def test_exact_real_preactivation_encloses_fp64_graph_to_tolerance(self):
        model=self.model
        with torch.no_grad():
            st=model.block(model.block(self.z1[None]))
            tok=model.block.tok_mix(st.transpose(1,2)).transpose(1,2)
            ln1=model.block.ln1(st+tok)
            truth=(ln1+model.block.chan_mix(ln1))[0]
        pre=pre_final_ln2(CertifiedBlock(model.block),self.z1)
        self.assertEqual((pre.nrows(),pre.ncols()),(81,48))
        for i,j in ((0,0),(0,47),(25,4),(80,40)):
            point=float(truth[i,j])
            v=Bound.from_arb(pre[i,j].str(42))
            self.assertLessEqual(float(v.lo)-2e-10,point)
            self.assertGreaterEqual(float(v.hi)+2e-10,point)

    def test_final_ln_shift_is_removed_numerically(self):
        model=self.model
        with torch.no_grad():
            st=model.block(model.block(self.z1[None]))
            tok=model.block.tok_mix(st.transpose(1,2)).transpose(1,2)
            ln1=model.block.ln1(st+tok)
            before=ln1+model.block.chan_mix(ln1)
            gauge=torch.arange(81,dtype=torch.float64).view(1,81,1)*Q_SCALE
            a=model.block.ln2(before)
            b=model.block.ln2(before+gauge)
            self.assertLess((a-b).abs().max().item(),1e-10)
            attack=before.clone();attack[0,3,0]+=0.1
            self.assertGreater((a-model.block.ln2(attack)).abs().max().item(),1e-8)

    def test_arb_serialized_centered_features_are_enclosing(self):
        pre=pre_final_ln2(CertifiedBlock(self.model.block),self.z1)
        strings=centered_arbs(pre.tolist()[0])
        self.assertEqual(len(strings),48)
        self.assertLess(abs(sum(float((x.lo+x.hi)/2) for x in map(Bound.from_arb,strings))),1e-8)


Q_SCALE=0.004
if __name__=='__main__':unittest.main()
