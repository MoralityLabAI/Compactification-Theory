import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from research.dimensional_atlas.gym.scm21_conservative_causal.core import (
    HERE,Model,Journal,OracleAudit,features,pair_pool,run,losses,digest,save_model,device_setup,put,Registry,bound_parameter_step,OOD)
from research.dimensional_atlas.gym.scm21_conservative_causal.campaign import train,choose,protected


class CausalConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((HERE/'config.json').read_text())
        self.c.update(hidden=16,heads=4,head_dim=8,sae_width=20,sae_k=3,batch=4,pair_batch=2,checkpoint_every=1,selection_steps=[1])
        device_setup('cpu',self.c);torch.manual_seed(2020)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.j=Journal(self.root/'work.jsonl',self.c)
        self.audit=OracleAudit(HERE.parents[4]/'EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py',self.j)
        self.oracle=self.audit.module
        self.pairs=pair_pool(self.oracle,2,120020,'unit_fixture',('signed',),set(),'coefficient')
        self.x=torch.stack([torch.stack([features(a),features(b)]) for a,b in self.pairs])
        self.model=Model(self.c,'coefficient');self.model.lock()
        for p in self.model.adapter.parameters():p.requires_grad_(True)
    def tearDown(self):self.j.file.close();self.temp.cleanup()
    def outputs(self,model):
        a=run(model,self.x[:,0]);b=run(model,self.x[:,1]);p=run(model,self.x[:,0],patch=(1,b['writes'][1],None))
        return a,b,p
    def test_exact_scm19_forward_and_no_inference_oracle(self):
        before=dict(self.audit.counts)
        ours=run(self.model,self.x[:,0])['logits'];old=self.model(self.x[:,0],policy='adapter')['logits']
        torch.testing.assert_close(ours,old,rtol=0,atol=0)
        self.assertEqual(before,dict(self.audit.counts))
    def test_live_donor_causal_gradient_and_frozen_protection(self):
        base=protected(self.model);a,b,p=self.outputs(self.model);b['writes'][1].retain_grad()
        mask=torch.zeros(2,64,dtype=torch.bool);mask[:,0]=True
        _,parts=losses(a,b,p,mask,mask,a['logits'].detach(),b['logits'].detach(),torch.ones(2),{**self.c,'causal_weight':1.})
        parts['causal_nll'].backward()
        self.assertGreater(float(b['writes'][1].grad.abs().max()),0)
        self.assertGreater(float(self.model.adapter.write.grad.abs().max()),0)
        self.assertEqual(protected(self.model),base)
        self.assertTrue(all(p.grad is None for n,p in self.model.named_parameters() if not n.startswith('adapter.')))
    def test_terminal_native_null_and_full_cls_positive(self):
        a=run(self.model,self.x[:,0],capture=True);b=run(self.model,self.x[:,1],capture=True)
        native=run(self.model,self.x[:,0],patch=(3,b['writes'][3],None))
        full=run(self.model,self.x[:,0],patch=(3,b['states'][3]-a['states'][3]))
        torch.testing.assert_close(native['logits'],a['logits'],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(full['logits'],b['logits'],atol=2e-6,rtol=2e-6)
    def test_lambda_zero_control_matches_math_distillation_gradient(self):
        other=copy.deepcopy(self.model);m=torch.zeros(2,64,dtype=torch.bool);m[:,2]=True
        gradients=[]
        for model,manual in ((self.model,False),(other,True)):
            a,b,p=self.outputs(model)
            total,parts=losses(a,b,p,m,m,torch.zeros(2,64),torch.zeros(2,64),torch.ones(2),{**self.c,'causal_weight':0.})
            loss=parts['clean_nll']+self.c['clean_kl_weight']*parts['clean_kl']+self.c['energy_weight']*parts['write_energy'] if manual else total
            loss.backward();gradients.append([p.grad.clone() for p in model.adapter.parameters()])
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    def test_generation_and_label_role_separation(self):
        a,b=self.pairs[0];self.assertTrue(a.case_id.startswith('scm21/'))
        self.assertEqual(a.source_graph[1:],b.source_graph[1:]);self.assertEqual(a.query,b.query)
        self.assertNotEqual(a.hidden_target,b.hidden_target)
        self.assertNotEqual(a.signature,b.signature)
        with self.assertRaises(RuntimeError):self.oracle.roster()
        before=dict(self.audit.counts)
        corrupted=__import__('dataclasses').replace(a,hidden_target=(999,),source_graph=('R','R','R'))
        torch.testing.assert_close(features(a),features(corrupted),rtol=0,atol=0)
        self.assertEqual(before,dict(self.audit.counts))
    def test_logger_kind_payload_and_json_receipts(self):
        self.j.event('unit_cell',kind='ood');self.j.checkpoint(self.audit)
        last=json.loads((self.root/'work.jsonl').read_text().splitlines()[-1]);self.assertEqual(last['kind'],'ood')
        put(self.root/'receipt.json',{'test':True});self.assertTrue(json.loads((self.root/'receipt.json').read_text())['test'])
        with self.assertRaises(FileExistsError):put(self.root/'receipt.json',{})
    def test_single_update_pipeline_and_snapshot(self):
        old=self.root/'old';old.mkdir();private=self.root/'private';private.mkdir();out=self.root/'out';out.mkdir()
        save_model(old/'frozen_s1.pt',self.model,{'encoding':'coefficient','seed':1,'steps':0,'locked':True})
        mask=torch.zeros(2,2,64,dtype=torch.bool);mask[:,:,0]=True
        teacher={'logits':torch.zeros(2,2,64),'eligible':torch.ones(2)}
        value=train(self.c,1,1.,1,{'x':self.x,'mask':mask},teacher,old,private,out,self.j,self.audit,{}, {},{},'cpu')
        self.assertTrue(value['base_and_sae_unchanged']);self.assertEqual(value['work']['model_calls'],3)
        self.assertEqual(value['work']['forward_examples'],6);self.assertEqual(value['work']['adapter_optimizer_steps'],1)
        saved=torch.load(private/'s1_lambda1.resume.pt',weights_only=True);self.assertEqual(saved['step'],1)
        self.assertIn('1',value['checkpoints'])
    def test_selection_gate_and_tie_rule(self):
        point={'step':1000,'clean_preserved':True,'mediation':{'id':{'native':.5,'random':0.,'eligible_common':64},'ood':{'native':.25,'random':0.,'eligible_common':64}}}
        candidate={'causal_weight':1.,'development':[point],'checkpoints':{'1000':{'file':'unit.pt'}}}
        self.assertTrue(choose(self.c,[candidate])['promoted'])
        candidate['development'][0]['clean_preserved']=False
        self.assertFalse(choose(self.c,[candidate])['promoted'])
    def test_encoded_query_duplicate_is_rejected(self):
        from dataclasses import replace
        a=self.pairs[0][0];registry=Registry();self.assertTrue(registry.accept((a,)))
        b=replace(a,case_id='different',query=tuple(x+1 for x in a.query))
        self.assertNotEqual(a.signature,b.signature)
        self.assertFalse(registry.accept((b,)))
        self.assertEqual(registry.rejections['prior_encoded'],1)
    def test_all_ood_family_topology_cells_are_balanced(self):
        from collections import Counter
        pairs=pair_pool(self.oracle,16,130000000,'unit_factorial',OOD,Registry(),'coefficient')
        counts=Counter((a.family,a.topology) for a,b in pairs)
        self.assertEqual(len(counts),16);self.assertEqual(set(counts.values()),{1})
    def test_hard_parameter_step_cap(self):
        params=list(self.model.adapter.parameters());before=[p.detach().clone() for p in params]
        with torch.no_grad():
            for p in params:p.add_(1.)
        stats=bound_parameter_step(params,before,self.c['max_parameter_step_l2'])
        self.assertTrue(stats['step_capped']);self.assertLessEqual(stats['step_l2'],.010001)
        self.assertGreater(stats['step_l2'],0.)


if __name__=='__main__':unittest.main()
