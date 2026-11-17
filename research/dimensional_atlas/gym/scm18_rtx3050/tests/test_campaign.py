import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch

from research.dimensional_atlas.gym.scm18_rtx3050.core import (
    HERE, MathModel, Meter, device_setup, digest, evaluate, fit_dictionary,
    load_checkpoint, checkpoint, operator_counterfactual, pool, train_base,
    train_adapter, replace_adapter, write_new)
from research.dimensional_atlas.gym.scm18_rtx3050.campaign import OracleAudit, mine, make_pools
from research.dimensional_atlas.gym.scm18_rtx3050.diagnostics import calibrate, compression_hook
from research.dimensional_atlas.gym.tasks import load_evalmath, make_evalmath_case


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((HERE/'config.json').read_text())
        self.c.update(hidden=16,sae_width=20,sae_k=3,readout_dim=4,
                      base_steps=2,sae_steps=2,adapter_steps=2,batch=4,
                      train_count=12,discovery_count=4,calibration_count=4,
                      confirmation_count=4,development_count=4)
        device_setup('cpu',self.c)
        torch.manual_seed(181)
        self.m=MathModel(self.c,4,2)
        self.oracle_path=HERE.parents[4]/'EvalMath/src/spiral_ln/algebraic_reasoning_oracle.py'
        if not self.oracle_path.exists():
            self.skipTest('separately pinned EvalMath source checkout required')
        self.o=load_evalmath(self.oracle_path)
        self.data,self.seen=make_pools(self.o,self.c)

    def test_no_split_overlap_and_official_access_fails_closed(self):
        all_sigs=[t.signature for ts in self.data.values() for t in ts]
        self.assertEqual(len(all_sigs),len(set(all_sigs)))
        audit=OracleAudit(self.o)
        with self.assertRaisesRegex(RuntimeError,'sealed'):
            self.o.roster(include_test=True)
        self.assertEqual(audit.counts['roster'],0)

    def test_exact_counterfactual_holds_series_and_other_operators(self):
        t=self.data['train'][0];cf=operator_counterfactual(t,self.o)
        self.assertEqual(t.query,cf.query)
        self.assertEqual([x for x,_ in t.demos],[x for x,_ in cf.demos])
        self.assertEqual(t.source_graph[1:],cf.source_graph[1:])
        self.assertNotEqual(t.source_graph[0],cf.source_graph[0])
        self.assertTrue(self.o.check_candidate(cf.oracle_case,cf.source_graph)['accepted'])

    def test_linear_fiber_null_zero_codes_and_frozen_base(self):
        train_base(self.m,self.data['train'],self.c,181)
        before=digest(self.m,('embed.','core.','positions','head.'))
        fit_dictionary(self.m,self.data['train'],self.c,181)
        replace_adapter(self.m,self.c,2,181)
        train_adapter(self.m,self.data['train'],self.c,181,'fiber_only','causal_consistency',self.o)
        self.assertEqual(before,digest(self.m,('embed.','core.','positions','head.')))
        x=torch.randn(2,4,30)
        zero=self.m(x,policy='zero_codes').logits
        base=self.m(x,policy='tied').logits
        self.assertTrue(torch.equal(zero,base))
        q=self.m.vpd.readout_basis
        v=torch.randn(16);v-=q@(q.T@v)
        changed=self.m(x,policy='joint',hook=lambda at,h:h+v if at==3 else h)
        clean=self.m(x,policy='joint')
        self.assertLess(float((changed.logits-clean.logits).abs().max()),2e-5)

    def test_cloned_untied_initial_compute_and_output_match(self):
        torch.manual_seed(12);tied=MathModel(self.c,4,2)
        torch.manual_seed(12);untied=MathModel(self.c,4,2,untied=True)
        x=torch.randn(2,4,30)
        self.assertTrue(torch.equal(tied(x).logits,untied(x).logits))
        self.assertGreater(sum(p.numel() for p in untied.parameters()),sum(p.numel() for p in tied.parameters()))
        meter=Meter(torch.device('cpu'),self.c);untied.meter=meter;untied(x)
        self.assertEqual(meter.counts['recurrent_example_visits'],8)

    def test_local_maximum_all_nine_neighbors_and_test_rejected(self):
        rows=mine(self.o,self.data['discovery'])
        for row in rows['rows']:
            maximum=row['maximum']
            self.assertEqual(len(maximum['neighbor_scores']),9)
            self.assertGreaterEqual(maximum['score'],max(maximum['neighbor_scores']))
        with self.assertRaises(ValueError):
            mine(self.o,self.data['train'])

    def test_top1_invalid_candidate_is_false_and_actual_checks_counted(self):
        train_base(self.m,self.data['train'],self.c,181)
        audit=OracleAudit(self.o)
        result=evaluate(self.m,self.data['development'],self.o,2)
        self.assertLessEqual(result['overall']['verifier_calls'],2*len(self.data['development']))
        self.assertGreaterEqual(audit.counts['check_candidate'],result['overall']['verifier_calls'])

    def test_full_pca_identity_checkpoint_roundtrip_and_no_overwrite(self):
        train_base(self.m,self.data['train'],self.c,181)
        frame=calibrate(self.m,self.data['calibration'])
        x=torch.randn(2,4,30)
        clean=self.m(x,policy='joint')
        projected=self.m(x,policy='joint',hook=compression_hook(self.m,frame,'pca',16,1))
        self.assertLess(float((projected.logits-clean.logits).abs().max()),2e-5)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'model.pt'
            ck=checkpoint(p,self.m,{'visits':4,'rank':2})
            other,_=load_checkpoint(p,self.c,torch.device('cpu'))
            self.assertEqual(digest(other),ck['model_sha256'])
            with self.assertRaises(FileExistsError):
                checkpoint(p,self.m,{})
            q=Path(directory)/'receipt.json';write_new(q,{'ok':True})
            with self.assertRaises(FileExistsError):
                write_new(q,{})


if __name__=='__main__':
    unittest.main()
