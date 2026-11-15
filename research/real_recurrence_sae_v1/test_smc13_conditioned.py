"""SMC-13C conditional feature alignment controls, independent of source teacher."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc13_conditioned_transfer import conditional_affinity,conditional_pair_shuffle,score_conditional_match


class ConditionalTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_public_digit_and_position_centering_erases_trivial_features(self):
        A=torch.zeros((8,3,96));B=torch.zeros_like(A)
        x=torch.tensor([[0,1,2],[0,1,2],[1,0,2],[1,0,2],
                        [0,1,2],[0,1,2],[1,0,2],[1,0,2]])
        for i in range(8):
            for pos in range(3):
                # Both features are completely determined by public clue.
                A[i,pos,0]=int(x[i,pos]==0)
                A[i,pos,1]=int(x[i,pos]!=0)
                B[i,pos,5]=int(x[i,pos]==0)
                B[i,pos,7]=int(x[i,pos]!=0)
        conditioned,info=conditional_affinity(A,B,x,'position_digit')
        self.assertEqual(info['calibration_puzzles'],8)
        self.assertEqual(info['positions'],3)
        self.assertTrue(all(abs(v)<1e-10 for row in conditioned for v in row))
        ordinary,_=conditional_affinity(A,B,x,'position')
        self.assertGreater(ordinary[0][5],0)

    def test_within_group_shuffle_preserves_each_marginal(self):
        torch.manual_seed(218)
        X=(torch.rand(12,4,96)<0.16).float()
        input_symbol=torch.randint(0,5,(12,4))
        Y=conditional_pair_shuffle(X,input_symbol,seed=27)
        for pos in range(4):
            for digit in range(5):
                mask=input_symbol[:,pos]==digit
                self.assertTrue(torch.equal(
                    X[mask,pos,:].sum(0),Y[mask,pos,:].sum(0)))
        self.assertTrue(torch.equal(
            Y,conditional_pair_shuffle(X,input_symbol,seed=27)))

    def test_same_feature_conditioned_pairing_outperforms_shuffled_calibration(self):
        torch.manual_seed(2217)
        # One independent latent feature varies WITHIN position/digit groups.
        X=(torch.rand(32,5,96)<0.2).float()
        inputs=torch.zeros((32,5),dtype=torch.long)
        Y=X.clone()
        Xtest=X[16:].clone();Ytest=Y[16:].clone()
        stats=score_conditional_match(X[:16],Y[:16],inputs[:16],
            {'toy':(Xtest,Ytest)},seed=77)
        for kind in ('position','position_digit'):
            self.assertGreater(stats[kind]['test_families']['toy']['heldout_overlap']['mean'],0.8)
            self.assertEqual(stats[kind]['calibration_group_counts']['positions'],5)

    def test_shape_and_metadata_guards(self):
        with self.assertRaises(ValueError):
            conditional_affinity(torch.zeros((2,3,96)),torch.zeros((3,3,96)),
                                 torch.zeros((2,3),dtype=torch.long))
        with self.assertRaises(ValueError):
            conditional_affinity(torch.zeros((2,3,96)),torch.zeros((2,3,96)),
                                 torch.zeros((2,3),dtype=torch.long),'unknown')
        with self.assertRaises(ValueError):
            conditional_pair_shuffle(torch.zeros((2,3,96)),torch.zeros((2,4),dtype=torch.long))


if __name__=='__main__':
    unittest.main()
