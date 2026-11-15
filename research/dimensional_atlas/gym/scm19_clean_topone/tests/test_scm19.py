import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from research.dimensional_atlas.gym.scm19_clean_topone.core import (
    HERE,PROGRAMS,Model,Journal,OracleAudit,features,groups,correct_mask,make_pool,
    counterfactual,set_loss,digest,save_model,load_model,device_setup)
from research.dimensional_atlas.gym.tasks import make_evalmath_case


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((HERE/'config.json').read_text());self.c.update(hidden=16,heads=4,head_dim=8,sae_width=20,sae_k=3)
        device_setup('cpu',self.c)
        self.tmp=tempfile.TemporaryDirectory();self.journal=Journal(Path(self.tmp.name)/'journal.jsonl',self.c)
        source=HERE.parents[4]/'EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py'
        self.audit=OracleAudit(source,self.journal);self.o=self.audit.module
    def tearDown(self):self.journal.file.close();self.tmp.cleanup()
    def case(self):return make_evalmath_case(self.o,9519,case_id='unit/9519',family='signed',topology='fork',graph=('I','I','R'))
    def test_public_features_ignore_gold_and_call_no_oracle(self):
        t=self.case();before=dict(self.audit.counts);x=features(t)
        other=replace(t,source_graph=('S','N','I'),hidden_target=tuple(-v for v in t.hidden_target))
        self.assertTrue(torch.equal(x,features(other)));self.assertEqual(before,dict(self.audit.counts))
        self.assertEqual(tuple(x.shape),(2,13,51));self.assertTrue(torch.isfinite(x).all())
    def test_verified_equivalent_labels_and_set_loss(self):
        t=self.case();g,_=groups(self.o);mask=correct_mask(t,self.o,g)
        self.assertTrue(mask[PROGRAMS.index(('I','I','R'))]);self.assertTrue(mask[PROGRAMS.index(('R','R','I'))])
        logits=torch.zeros(1,64,requires_grad=True);loss=set_loss(logits,mask[None]);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),float(torch.log(torch.tensor(64./int(mask.sum())))),places=5)
        with self.assertRaises(ValueError):set_loss(logits,torch.zeros_like(mask)[None])
    def test_common_initialization_and_predictor_no_oracle(self):
        torch.manual_seed(191);a=Model(self.c,'coefficient')
        torch.manual_seed(191);b=Model(self.c,'polynomial_gram')
        prefix=('positions','core.','head.','sae.','adapter.')
        self.assertEqual(digest(a,prefix),digest(b,prefix))
        t=self.case();x=features(t)[None];before=dict(self.audit.counts)
        out=b(x);self.assertEqual(tuple(out['logits'].shape),(1,64));self.assertEqual(before,dict(self.audit.counts))
        set_loss(out['logits'],torch.ones(1,64,dtype=torch.bool)).backward()
    def test_first_operator_counterfactual_holds_inputs_other_ops(self):
        a=self.case();b=counterfactual(a,self.o)
        self.assertEqual(a.query,b.query);self.assertEqual([x for x,_ in a.demos],[x for x,_ in b.demos])
        self.assertEqual(a.source_graph[1:],b.source_graph[1:]);self.assertNotEqual(a.source_graph[0],b.source_graph[0])
        self.assertTrue(self.o.check_candidate(b.oracle_case,b.source_graph)['accepted'])
    def test_head_fiber_terminal_null_and_sham(self):
        torch.manual_seed(19);m=Model(self.c,'coefficient');m.lock()
        x=features(self.case())[None];clean=m(x,policy='adapter')['logits']
        q=m.adapter.q;v=torch.randn(1,16);v=v-(v@q)@q.T
        def hook(at,h):
            if at==3:h=h.clone();h[:,0]+=v
            return h
        changed=m(x,policy='adapter',hook=hook)['logits']
        self.assertLess(float((changed-clean).abs().max()),2e-5)
        self.assertTrue(torch.equal(m(x,policy='base')['logits'],m(x,policy='zero')['logits']))
    def test_source_pin_roster_and_disjoint_pools(self):
        with self.assertRaisesRegex(RuntimeError,'forbidden'):self.o.roster()
        seen=set();a=make_pool(self.o,8,31000000,'a',('signed',),seen);b=make_pool(self.o,8,32000000,'b',('signed',),seen)
        self.assertFalse({x.signature for x in a}&{x.signature for x in b})
    def test_snapshot_roundtrip_and_no_overwrite(self):
        m=Model(self.c,'coefficient');m.lock();p=Path(self.tmp.name)/'model.pt'
        ck=save_model(p,m,{'encoding':'coefficient','locked':True});other=load_model(p,self.c,torch.device('cpu'))
        self.assertEqual(ck['model_sha256'],digest(other))
        with self.assertRaises(FileExistsError):save_model(p,m,{})


if __name__=='__main__':unittest.main()
