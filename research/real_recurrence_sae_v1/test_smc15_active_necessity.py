"""CPU-only unit checks for SMC-15, independent of unpublished checkpoint files."""
import copy
import sys
import unittest
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc15_active_necessity as s


class FakeSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(48,96)
        self.decoder=nn.Linear(96,48)
        with torch.no_grad():
            self.encoder.weight.zero_();self.encoder.bias.fill_(-10)
            for i in range(4):
                self.encoder.weight[i,i]=1
                self.encoder.bias[i]=0
            self.decoder.weight.zero_()
            for i in range(48):self.decoder.weight[i,i]=1
            for i in range(48,96):self.decoder.weight[i%48,i]=1
            self.decoder.bias.zero_()
    def encode(self,x):
        z=torch.relu(self.encoder(x))
        top=torch.topk(z,4,dim=-1)
        return torch.zeros_like(z).scatter(-1,top.indices,top.values)


class LinearTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Identity()
        self.head=nn.Linear(48,10,bias=False,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.zero_();self.head.weight[1,0]=2.


class NativeNecessityTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_fixed_top_four_physical_decoder_and_other_active_dose(self):
        sae=FakeSAE()
        stage=torch.ones((2,81,48),dtype=torch.float32)
        stage[:,:,:4]=torch.tensor([5.,4.,3.,2.])
        pos=torch.arange(8).repeat(2,1)
        mu=torch.zeros(48);scale=torch.ones(48)
        z,ids,strict=s.encoded_goals(stage,pos,sae,mu,scale)
        self.assertTrue(bool(strict.all()))
        self.assertTrue(bool((ids==0).all()))
        cols=s.physical_columns(sae,scale)
        self.assertEqual(tuple(cols.shape),(96,48))
        top_delta,coeff=s.feature_deltas(z,ids,cols)
        self.assertTrue(bool(torch.allclose(coeff,torch.full_like(coeff,5))))
        alt,mask=s.other_active_ids(z,ids,1,'fake',[25,26])
        self.assertTrue(bool(mask.all()))
        self.assertTrue(bool((alt>0).all()))
        alt2,mask2=s.other_active_ids(z,ids,1,'fake',[25,26])
        self.assertTrue(torch.equal(alt,alt2))
        control=s.dose_matched_other(top_delta,alt,cols)
        self.assertTrue(bool(torch.allclose(control.norm(dim=-1),top_delta.norm(dim=-1))))
        self.assertFalse(bool(torch.any((control*top_delta).sum(-1)!=0)))

    def test_native_erasure_is_single_goal_and_changes_correct_margin(self):
        model=LinearTeacher().eval()
        stage=torch.zeros((1,81,48),dtype=torch.float32)
        stage[:,:,0]=1.
        pos=torch.arange(8)[None,:]
        y=torch.ones((1,8),dtype=torch.long)
        clean=model.head(stage.double())
        delta=torch.zeros((1,8,48),dtype=torch.float64)
        delta[:,:,0]=1.5
        drops,correct,flips=s.ablate_margin(model,stage,clean,y,pos,delta)
        self.assertTrue(bool(correct.all()))
        self.assertTrue(bool(flips.all()))
        self.assertTrue(bool(torch.allclose(drops,torch.full_like(drops,3.))))
        zero,_,flips0=s.ablate_margin(model,stage,clean,y,pos,torch.zeros_like(delta))
        self.assertTrue(bool((zero==0).all()))
        self.assertFalse(bool(flips0.any()))

    def test_physical_erasure_reencodes_after_subtraction(self):
        sae=FakeSAE();m=torch.zeros(48);scale=torch.ones(48)
        stage=torch.zeros((1,81,48),dtype=torch.float32)
        stage[:,:,0]=3.;stage[:,:,1]=2.;stage[:,:,2]=1.5;stage[:,:,3]=1.
        pos=torch.arange(8)[None,:]
        z,top,_=s.encoded_goals(stage,pos,sae,m,scale)
        delta,_=s.feature_deltas(z,top,s.physical_columns(sae,scale))
        after=s.reencoded_matched(stage,pos,sae,m,scale,delta,top)
        self.assertTrue(bool((after==0).all()))

    def test_calibration_sign_and_puzzle_cluster_bootstrap(self):
        a=torch.tensor([[1.,2.,-3.,4.,5.,6.,7.,8.]],dtype=torch.float64)
        m=-a;c=a.clone();mask=torch.ones_like(a,dtype=torch.bool)
        orientation=s.orient(a,m,c,mask)
        self.assertEqual(orientation['matched']['sign'],-1)
        self.assertEqual(orientation['other']['sign'],1)
        self.assertTrue(orientation['matched']['identified'])
        rows=[]
        for p in range(64):
            mask=[p<20 and i<2 for i in range(8)]
            rows.append({'eligible_mask':mask,'source_drop':[1.]*8,
                         'matched_drop':[1.]*8,'other_active_drop':[-1.]*8})
        oriented={'matched':{'sign':1,'identified':True},
                  'other':{'sign':1,'identified':True}}
        score=s.score_rows(rows,oriented,123)
        self.assertTrue(score['registered_success'])
        self.assertEqual(score['eligible_goals'],40)
        self.assertEqual(score['nonempty_puzzle_clusters'],20)
        self.assertAlmostEqual(score['matched_minus_other_active'],2.0)
        fail=s.score_rows(rows,{'matched':{'sign':1,'identified':False},'other':oriented['other']},123)
        self.assertFalse(fail['registered_success'])

    def test_inactive_target_feature_never_selects_active_control(self):
        z=torch.zeros((1,8,96))
        z[:,:,1]=2;z[:,:,2]=1
        match=torch.zeros((1,8),dtype=torch.long)
        control,eligible=s.other_active_ids(z,match,0,'fake',[33])
        self.assertFalse(bool(eligible.any()))
        self.assertTrue(bool((control==0).all()))

    def test_invalid_geometry_abstains(self):
        with self.assertRaises(ValueError):s.fresh_rosters({},limit=5)
        with self.assertRaises(ValueError):s.feature_deltas(torch.zeros(1,8,96),torch.zeros(1,4,dtype=torch.long),torch.zeros(96,48))
        with self.assertRaises(ValueError):s.ablate_margin(LinearTeacher(),torch.zeros(1,81,48),torch.zeros(1,81,10),torch.ones(1,8,dtype=torch.long),torch.arange(8)[None,:],torch.zeros(1,8,48))


if __name__=='__main__':unittest.main()
