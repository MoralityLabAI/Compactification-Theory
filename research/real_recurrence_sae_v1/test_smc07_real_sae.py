"""SMC-07 CPU unit controls, using an untrained toy recurrent block.

These tests exercise the experiment harness without fetching external weights
or claiming that the toy model supplies scientific evidence.
"""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc07_real_sae import (
    TopKSAE,split_ids,fit_sae,pca_basis,reconstruction,
    sparsity_and_error,goal_replay,feature_transport,
)

class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(48)
        self.ln2=nn.LayerNorm(48)
        self.tok_mix=nn.Sequential(nn.Linear(81,16),nn.GELU(),nn.Linear(16,81))
        self.chan_mix=nn.Sequential(nn.Linear(48,24),nn.GELU(),nn.Linear(24,48))
    def forward(self,z):
        z=self.ln1(z+self.tok_mix(z.transpose(1,2)).transpose(1,2))
        return self.ln2(z+self.chan_mix(z))

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok=nn.Embedding(10,48)
        self.pos=nn.Embedding(81,48)
        self.block=ToyBlock()
        self.head=nn.Linear(48,10)
    def reason_states(self,x):
        pos=torch.arange(81).unsqueeze(0).expand(x.shape[0],-1)
        h=self.tok(x)+self.pos(pos)
        states=[h]
        for _ in range(4):
            h=self.block(h)
            states.append(h)
        return states

class SMC07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(27)
        torch.set_num_threads(1)
        cls.model=Toy().eval()
        cls.inputs=torch.randint(1,10,(8,81))
        cls.inputs[:,3]=0
        cls.targets=torch.randint(1,10,(8,81))
        cls.split=split_ids(8)
        with torch.no_grad(): cls.states=cls.model.reason_states(cls.inputs)

    def test_grouped_split(self):
        self.assertEqual(self.split,{'train':[0,1,4,5],'val':[2,6],'test':[3,7]})
        with self.assertRaises(ValueError): split_ids(6)

    def test_sae_topk_training_and_pca4(self):
        s,mu,std,fit=fit_sae(self.states[1][self.split['train']],
                             self.states[1][self.split['val']],steps=20,
                             k=4,width=96,seed=17)
        self.assertEqual(s.encoder.out_features,96)
        self.assertLessEqual(fit['best_step'],20)
        with torch.no_grad():
            x=(self.states[1][self.split['test']].reshape(-1,48)-mu)/std
            z=s.encode(x)
        self.assertTrue(bool((z>0).sum(-1).le(4).all()))
        basis,_=pca_basis(self.states[1][self.split['train']],mu,std)
        self.assertEqual(tuple(basis.shape),(48,4))
        rec=reconstruction(s,self.states[1][3,3],mu,std)
        self.assertEqual(tuple(rec.shape),(48,))
        stats=sparsity_and_error(s,self.states[1][self.split['test']],mu,std,
                                  {'pca4':basis})
        self.assertGreaterEqual(stats['sae_mse_normalized'],0)
        transport=feature_transport(s,self.states,self.split,mu,std)
        self.assertEqual(transport['n_token_pairs'],162)

    def test_paired_causal_replay_and_fp64_layernorm_gauge(self):
        s,mu,std,_=fit_sae(self.states[1][self.split['train']],
                           self.states[1][self.split['val']],steps=20,seed=17)
        basis,_=pca_basis(self.states[1][self.split['train']],mu,std)
        result=goal_replay(self.model,self.states,self.inputs,self.targets,
                            self.split,s,mu,std,basis,seed=17)
        self.assertEqual(result['n_independent_test_puzzles'],2)
        self.assertEqual(len(result['gauge_checks_fp64']),2)
        for row in result['gauge_checks_fp64']:
            self.assertLess(row['gauge_logit_linf'],1e-9)
            self.assertGreater(row['transverse_logit_linf'],1e-8)

if __name__=='__main__': unittest.main()
