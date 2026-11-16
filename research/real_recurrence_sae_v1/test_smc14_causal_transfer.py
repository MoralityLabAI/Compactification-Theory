"""SMC-14 CPU tests for source-independent intervention and null mechanics."""
import copy
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc14_causal_transfer import (
    GOALS,goal_positions,fixed_competitors,signed_margin_response,
    centered_cosine,calibration_orientation,interval_bootstrap,
    within_puzzle_shuffle_null,load_conditioned,puzzle_case_scores,
    summarize_family,physical_decoder_unit_table
)
from smc07_real_sae import TopKSAE


class LinearToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
        self.head=nn.Linear(48,10,bias=False,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1,0]=2.
            self.head.weight[2,1]=-1.


class CausalTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_goal_selection_uses_visible_missing_positions_only(self):
        x=torch.ones(2,81,dtype=torch.long)
        x[0,torch.arange(0,30)]=0
        x[1,torch.arange(10,44)]=0
        pos=goal_positions(x)
        self.assertEqual(tuple(pos.shape),(2,GOALS))
        self.assertEqual(pos[0].tolist(),[0,3,7,11,15,18,22,26])
        self.assertEqual(pos[1,0].item(),10)
        self.assertTrue(bool((x.gather(1,pos)==0).all()))
        with self.assertRaises(ValueError):
            goal_positions(torch.ones(1,81,dtype=torch.long))

    def test_two_sided_signed_response_matches_linear_teacher_exactly(self):
        model=LinearToyTeacher().double().eval()
        stage=torch.zeros((2,81,48),dtype=torch.float32)
        inputs=torch.ones(2,81,dtype=torch.long)
        inputs[:,:30]=0
        positions=goal_positions(inputs)
        truth=torch.ones(2,GOALS,dtype=torch.long)
        feature_id=torch.zeros(2,GOALS,dtype=torch.long)
        table=torch.zeros(96,48,dtype=torch.float64)
        table[:,0]=1.
        clean=model.head(stage.double())
        d,correct=signed_margin_response(model,stage,clean,truth,positions,
                                          feature_id,table,radius=0.2)
        self.assertEqual(tuple(d.shape),(2,GOALS))
        self.assertTrue(bool(torch.allclose(d,torch.full_like(d,2),atol=1e-12)))
        self.assertFalse(bool(correct.any()))
        self.assertEqual(fixed_competitors(clean[0,positions[0]],truth[0]).tolist(),
                         [0]*GOALS)

    def test_calibration_sign_and_nuisance_centered_cosine(self):
        a=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.],dtype=torch.float64)
        b=-a*3+13
        self.assertAlmostEqual(centered_cosine(a,b),-1.)
        self.assertAlmostEqual(centered_cosine(a+24,b),-1.)
        cal=calibration_orientation(torch.stack((a,a)),torch.stack((b,b)),
                                    torch.ones((2,GOALS),dtype=torch.bool))
        self.assertEqual(cal['sign'],-1)
        self.assertAlmostEqual(centered_cosine(a,cal['sign']*b),1.)
        self.assertIsNone(centered_cosine(torch.ones(GOALS),b))
        with self.assertRaises(ValueError):
            centered_cosine(a,b[:6])
        bad=calibration_orientation(torch.zeros(2,GOALS),torch.ones(2,GOALS),
                                    torch.ones((2,GOALS),dtype=torch.bool))
        self.assertTrue(bad['unidentified'])

    def test_bootstrap_puzzle_clusters_and_calibration_null(self):
        stats=interval_bootstrap([0.3]*32,seed=88,draws=200)
        self.assertAlmostEqual(stats['mean'],0.3)
        self.assertEqual(stats['n_independent_puzzles'],32)
        self.assertEqual(len(stats['puzzle_bootstrap95']),2)
        a=torch.arange(GOALS,dtype=torch.float64)
        r1=within_puzzle_shuffle_null(a,a,torch.ones(GOALS,dtype=torch.bool),
                                       1,6,11)
        r2=within_puzzle_shuffle_null(a,a,torch.ones(GOALS,dtype=torch.bool),
                                       1,6,11)
        self.assertEqual(r1,r2)
        self.assertLess(r1,0.95)

    def test_frozen_physical_decoder_norm_and_invalid_conditions(self):
        torch.manual_seed(8)
        sae=TopKSAE(48,96,4)
        scale=torch.rand(48)+0.4
        table=physical_decoder_unit_table(sae,scale)
        self.assertEqual(tuple(table.shape),(96,48))
        self.assertEqual(table.dtype,torch.float64)
        self.assertTrue(bool(torch.allclose(table.norm(dim=-1),
                    torch.ones(96,dtype=torch.float64),atol=1e-12)))
        with self.assertRaises(ValueError):
            fixed_competitors(torch.ones((2,9)),torch.ones(2,dtype=torch.long))

    def test_source_conditioned_receipt_hash_rejects_substitution(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'conditioned.json'
            path.write_text('{"schema":"plausible","results":[]}')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                load_conditioned(path)


if __name__=='__main__':
    unittest.main()
