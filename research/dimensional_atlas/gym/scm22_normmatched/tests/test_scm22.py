"""SCM-22 CPU safety, oracle and intervention geometry checks."""
import copy
import math
import sys
from pathlib import Path
import unittest
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from model import Case,P,exact,input_features,make_pairs,LoopedMath,train,checkpoint_digest,sample_cases
from geometry import readout_fiber,matched_patch,frame,fit_sae,frozen_states,make_frames
from run import sign_test,evaluate

class SCM22(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model,cls.training=train(991,updates=300)
        cls.sae,cls.info=fit_sae(cls.model,frozen_states(cls.model,sample_cases(128,1099)),991,steps=30)
        cls.cal=make_pairs(40,1111)
        cls.frames,cls.frameinfo=make_frames(cls.model,cls.sae,cls.info,cls.cal,991)
    def test_exact_math_oracle(self):
        for a in range(4):
            for b in range(4):
                for x in range(11):
                    c=Case(a,b,x,0)
                    y=x
                    for fac,offset in (((1,1),(1,3),(2,1),(3,2))[a],((1,2),(2,4),(3,1),(4,3))[b]):
                        y=(fac*y+offset)%11
                    self.assertEqual(exact(c),y)
                    self.assertTrue(0<=y<11)
    def test_pair_generation_strict_semantic_and_unique(self):
        p=make_pairs(256,123)
        self.assertEqual(len(set((a.first,b.first,a.second,a.x,a.nuisance) for a,b in p)),256)
        for a,b in p:
            self.assertEqual((a.second,a.x,a.nuisance),(b.second,b.x,b.nuisance))
            self.assertNotEqual(a.first,b.first)
            self.assertNotEqual(exact(a),exact(b))
    def test_recurrent_block_is_tied(self):
        self.assertEqual(self.model.visits,4)
        self.assertEqual(self.model.shared.qkv.weight.shape[0],48*3)
        h=input_features([Case(0,1,2,4)])
        z,states=self.model(h,record=True)
        self.assertEqual(tuple(z.shape),(1,11))
        self.assertEqual(len(states),4)
        self.assertEqual(tuple(states[-1].shape),(1,48))
    def test_pretrained_readout_fiber_is_linear_only(self):
        q=readout_fiber(self.model)
        self.assertEqual(tuple(q.shape),(48,37))
        self.assertLess(float((self.model.head.weight@q).abs().max()),1e-5)
        self.assertEqual(float((q.T@q-torch.eye(37)).abs().max())<1e-5,True)
    def test_sae_decoders_are_sparse(self):
        h=frozen_states(self.model,sample_cases(8,1088))
        _,c=self.sae(h.flatten(0,1)/h.flatten(0,1).square().mean(-1,keepdim=True).sqrt())
        self.assertLessEqual(int((c>0).sum(dim=-1).max()),8)
        self.assertTrue(torch.isfinite(c).all())
    def test_frames_orthonormal_and_label_blind(self):
        for name,q in self.frames.items():
            self.assertEqual(q.shape,(48,4))
            torch.testing.assert_close(q.T@q,torch.eye(4),atol=2e-5,rtol=0)
        self.assertEqual(self.frameinfo['calibration_pairs'],40)
        self.assertEqual(len(self.frameinfo['selected_sae_feature_ids']),4)
    def test_norm_matching_and_projection_energy(self):
        q=self.frames['difference_pca4'];d=torch.randn(17,48)
        for factor in (0.5,1.0):
            patch,ratio,fallbacks=matched_patch(d,q,factor)
            self.assertLess(float((patch.norm(dim=1)-d.norm(dim=1)*factor).abs().max()),1e-5)
            self.assertLessEqual(float(ratio.max()),1.00001)
            self.assertEqual(fallbacks,0)
    def test_patch_changes_only_one_visit(self):
        x=input_features([Case(0,1,2,3)])
        _,states=self.model(x,record=True)
        patch=torch.randn(1,48)*.1
        z,a=self.model(x,patch=(1,patch),record=True)
        torch.testing.assert_close(a[0],states[0],atol=0,rtol=0)
        torch.testing.assert_close(a[1],states[1]+patch,atol=1e-7,rtol=0)
    def test_exact_sign_test_edges(self):
        self.assertEqual(sign_test(0,0),1.)
        self.assertEqual(sign_test(4,0),.125)
        self.assertEqual(sign_test(3,3),1.)
    def test_checkpoint_digest_stability(self):
        a=checkpoint_digest(self.model)
        self.assertEqual(a,checkpoint_digest(self.model))
        modified=copy.deepcopy(self.model)
        with torch.no_grad():modified.head.bias[0]+=.01
        self.assertNotEqual(a,checkpoint_digest(modified))
    def test_refuse_insufficient_clean_eligible(self):
        with self.assertRaisesRegex(ValueError,'below 64'):
            evaluate(self.model,make_pairs(8,321),1,self.frames)

if __name__=='__main__':unittest.main()
