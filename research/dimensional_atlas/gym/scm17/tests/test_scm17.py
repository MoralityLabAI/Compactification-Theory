"""SCM-17 explicit RMP visit-hook contract and synthetic readout-fiber tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from research.dimensional_atlas.gym.scm17 import campaign
from research.dimensional_atlas.gym.scm17.rmp_fiber import (
  SAEVisitFiber,calibrate_decision_frame,causal_probe,run_with_adapter,tensor_digest)

class FakeDictionary(nn.Module):
    def __init__(self,hidden=6,width=16,k=3):
        super().__init__()
        self.decoder=nn.Parameter(torch.randn(width,hidden)*.15)
        self.encoder=nn.Parameter(torch.randn(hidden,width)*.15)
        self.k=k
    def encode(self,x):
        raw=x@self.encoder
        values,index=raw.topk(self.k,dim=-1)
        return torch.zeros_like(raw).scatter(-1,index,F.relu(values))

class FakeRMP(nn.Module):
    """Runs RMP-style visit hooks, final LayerNorm and a tied unembedding."""
    def __init__(self,hidden=6,vocab=16,visits=4,inject=False):
        super().__init__()
        self.expanded_visits=visits
        self.hidden_size=hidden
        self.emb=nn.Embedding(vocab,hidden)
        self.core=nn.Linear(hidden,hidden,bias=False)
        self.n1=nn.LayerNorm(hidden)
        self.n2=nn.LayerNorm(hidden)
        self.inject=inject
        with torch.no_grad():
            self.core.weight.copy_(torch.randn(hidden,hidden)*.34+torch.eye(hidden)*.49)
    def lens(self,x):
        return F.linear(self.n2(x),self.emb.weight)
    def run(self,tokens,*hooks):
        h=self.emb(tokens); base=h.clone()
        states=[]
        for i in range(self.expanded_visits):
            after=h+.24*F.gelu(self.core(self.n1(h)))
            if self.inject:after=after+base
            for hook in hooks:
                patch=hook(SimpleNamespace(visit=i,after=after,before=h))
                if patch is not None:after=patch
            h=after;states.append(h)
        return SimpleNamespace(logits=self.lens(h),visit_outputs=tuple(states))

class FakeAnalysis:
    def __init__(self,seed=7,inject=False):
        torch.manual_seed(seed)
        self.model=FakeRMP(inject=inject)
    def run(self,tokens,*hooks):
        return self.model.run(tokens,*hooks)

class FiberTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rmp=FakeAnalysis()
        self.tokens=torch.tensor([[1,5,3,4,7],[2,6,7,1,8],[2,3,5,0,10],[4,1,6,9,11]])
        self.frame=calibrate_decision_frame(self.rmp,self.tokens[:1],contrasts=1,max_rank=1)

    def test_local_basis_orthonormal(self):
        q=self.frame.q
        self.assertEqual(q.shape,(6,1))
        self.assertTrue(torch.allclose(q.T@q,torch.eye(1),atol=1e-5))
        self.assertAlmostEqual(self.frame.captured_gradient_energy,1.,places=5)
        self.assertEqual(self.frame.total_gradients,1)

    def test_fiber_is_orthogonal_to_selected_gradient(self):
        h=self.rmp.run(self.tokens[:1]).visit_outputs[-1][0,-1].detach().requires_grad_(True)
        logits=self.rmp.model.lens(h.view(1,1,-1))[0,0]
        order=logits.argsort(descending=True)
        grad=torch.autograd.grad(logits[order[0]]-logits[order[1]],h)[0]
        vec=torch.randn_like(grad)
        fiber=vec-(vec@self.frame.q)@self.frame.q.T
        self.assertLess(abs(float(fiber@grad)),2e-5)

    def test_final_local_margin_first_order_null(self):
        probe=causal_probe(self.rmp,self.tokens[:1],self.frame,visit_count=4,epsilon=1e-3)
        self.assertLess(abs(probe['central_slope_by_visit'][-1]),3e-3)
        self.assertEqual(probe['status'],'finite_float_causal_probe_only')
        self.assertEqual(len(probe['central_slope_by_visit']),4)

    def test_earlier_fiber_transport_can_be_visible(self):
        probe=causal_probe(self.rmp,self.tokens[:1],self.frame,visit_count=4,epsilon=2e-3)
        self.assertGreater(max(abs(x) for x in probe['central_slope_by_visit'][:-1]),
                           abs(probe['central_slope_by_visit'][-1])+1e-4)

    def test_trained_dictionary_is_frozen(self):
        sae=FakeDictionary()
        adapter=SAEVisitFiber(sae,self.frame,rank=3,visits=4)
        self.assertFalse(any(p.requires_grad for p in sae.parameters()))
        before=campaign.state_digest(self.rmp.model)
        targets=torch.tensor([1,4,2,3])
        out=campaign.train_adapter(self.rmp,adapter,self.tokens,targets,steps=4,
                 seed=3,batch_size=2,learning_rate=.002,device='cpu')
        self.assertEqual(out['steps'],4)
        self.assertEqual(before,campaign.state_digest(self.rmp.model))
        self.assertTrue(any(p.grad is not None for n,p in adapter.named_parameters()
                            if p.requires_grad))

    def test_adapter_controls_and_shuffled_visits(self):
        adapter=SAEVisitFiber(FakeDictionary(),self.frame,rank=3,visits=4)
        with torch.no_grad():
            clean,_=run_with_adapter(self.rmp,self.tokens,adapter=None)
            joint,e=run_with_adapter(self.rmp,self.tokens,adapter)
            visible,_=run_with_adapter(self.rmp,self.tokens,adapter,sector='visible_only')
            fiber,_=run_with_adapter(self.rmp,self.tokens,adapter,sector='fiber_only')
            zero,_=run_with_adapter(self.rmp,self.tokens,adapter,zero_codes=True)
            shuffled,_=run_with_adapter(self.rmp,self.tokens,adapter,visit_order=(3,2,1,0))
        self.assertEqual(len(e),4)
        self.assertEqual(visible.logits.shape,clean.logits.shape)
        self.assertEqual(fiber.logits.shape,clean.logits.shape)
        self.assertEqual(shuffled.logits.shape,clean.logits.shape)
        self.assertTrue(torch.allclose(zero.logits,clean.logits,atol=1e-6))
        self.assertGreater(float((joint.logits-clean.logits).abs().max()),0)

    def test_ffu_like_reinjection_order_preserved(self):
        analysis=FakeAnalysis(seed=9,inject=True)
        frame=calibrate_decision_frame(analysis,self.tokens[:1],contrasts=1,max_rank=1)
        adapter=SAEVisitFiber(FakeDictionary(),frame,rank=2,visits=4)
        with torch.no_grad():
            a,_=run_with_adapter(analysis,self.tokens,adapter)
            b=analysis.run(self.tokens)
        self.assertEqual(a.logits.shape,b.logits.shape)
        self.assertEqual(len(a.visit_outputs),4)

    def test_missing_artifacts_status(self):
        with tempfile.TemporaryDirectory() as d:
            x=Path(d)
            a=SimpleNamespace(model='LOOP-T s1',rmp_root=x/'src',runs=x/'runs',artifacts=x/'a',
                  monitor_bundle=x/'bundle',metrics=x/'dictionaries.json',dictionary=x/'D.pt')
            report=campaign.source_manifest(a)
            self.assertEqual(report['status'],'blocked_external_artifacts')
            self.assertIn('checkpoint_integrity_receipt',report['missing'])
            self.assertTrue(report['scientific_status']['unopened_sealed_test'])

    def test_balanced_group_disjoint_selection(self):
        data={'families':[],'groups':[],'ids':[]}
        for f in campaign.CORE:
            for j in range(30):
                data['families'].append(f);data['groups'].append(f'{f}/{j}');data['ids'].append(j)
        x=campaign.choose_rows(data,size=12,selection=None,seed=5)
        y=campaign.choose_rows(data,size=12,selection=False,seed=5)
        self.assertEqual(len(x),12)
        self.assertEqual(len(set(data['groups'][k] for k in x)),12)
        self.assertEqual(len(y),12)
        self.assertEqual({f:sum(data['families'][k]==f for k in x) for f in campaign.CORE},
                         {f:3 for f in campaign.CORE})

    def test_readout_honest_nonlinear_scope(self):
        self.assertIn('calibration_only',self.frame.selection)
        self.assertLess(self.frame.q.shape[1],self.rmp.model.hidden_size)

    def test_provenance_hash_shape_sensitive(self):
        a=torch.arange(6,dtype=torch.float32)
        self.assertNotEqual(tensor_digest(a),tensor_digest(a.reshape(2,3)))

    def test_unmatched_visit_rejected(self):
        adapter=SAEVisitFiber(FakeDictionary(),self.frame,rank=2,visits=4)
        with self.assertRaisesRegex(ValueError,'permutation'):
            run_with_adapter(self.rmp,self.tokens,adapter,visit_order=(0,1,1,2))

    def test_invalid_sector_rejected(self):
        adapter=SAEVisitFiber(FakeDictionary(),self.frame,rank=2,visits=4)
        with self.assertRaisesRegex(ValueError,'invalid sector'):
            adapter(torch.randn(1,5,6),0,sector='bad')

    def test_source_grading_requires_target_labels_separately(self):
        res=campaign.score(self.rmp,self.tokens,torch.tensor([1,2,3,4]))
        self.assertEqual(res['n'],4)
        self.assertEqual(res['model_forward_visits'],16)
        self.assertEqual(len(res['predictions_sha256']),64)

if __name__=='__main__':unittest.main()

class SourceLockTests(unittest.TestCase):
    def _synthetic_artifact_tree(self,root):
        root=Path(root)
        cell,stage=campaign.CELLS['LOOP-T s1']
        source=root/'rmp'; art=root/'artifacts'; runs=root/'runs'; bun=root/'bundle'
        bun.mkdir()
        for item in campaign.RMP_FILES:
            f=source/item
            f.parent.mkdir(parents=True,exist_ok=True)
            f.write_text('mock source: '+item)
        ck=art/cell/'checkpoints'/f'{stage}.pt'
        ck.parent.mkdir(parents=True,exist_ok=True);ck.write_bytes(b'mock-checkpoint')
        receipt={'sha256':campaign.file_hash(ck),'bytes':ck.stat().st_size}
        ck.with_name(ck.name+'.receipt.json').write_text(__import__('json').dumps(receipt))
        cfg=runs/cell/'result.json';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text('{}')
        metrics=root/'metrics.json';metrics.write_text('{}')
        dictionary=root/'dict.pt';dictionary.write_bytes(b'mock sae')
        return SimpleNamespace(model='LOOP-T s1',rmp_root=source,runs=runs,artifacts=art,
               monitor_bundle=bun,metrics=metrics,dictionary=dictionary)

    def test_hash_receipt_enforced_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            args=self._synthetic_artifact_tree(d)
            ready=campaign.source_manifest(args)
            self.assertEqual(ready['status'],'ready')
            ck=Path(ready['source']['checkpoint'])
            ck.write_bytes(b'mutated-checkpoint')
            blocked=campaign.source_manifest(args)
            self.assertEqual(blocked['status'],'blocked_external_artifacts')
            self.assertIn('checkpoint_integrity_mismatch',blocked['missing'])

    def test_freeze_source_manifest_rejects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            args=self._synthetic_artifact_tree(d)
            ready=campaign.source_manifest(args)
            import json
            manifest=Path(d)/'locked.json'
            manifest.write_text(json.dumps(ready))
            self.assertEqual(json.loads(manifest.read_text()),campaign.source_manifest(args))
            args.metrics.write_text('{"modified":true}')
            self.assertNotEqual(json.loads(manifest.read_text()),campaign.source_manifest(args))

    def test_trained_mode_requires_lock(self):
        with tempfile.TemporaryDirectory() as d:
            args=self._synthetic_artifact_tree(d)
            args.output=Path(d)/'out.json'
            args.device='cpu'
            args.lock_file=None
            with self.assertRaisesRegex(ValueError,'--lock-file'):
                campaign.execute(args)
