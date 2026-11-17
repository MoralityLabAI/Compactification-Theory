"""SCM-15 deterministic CPU unit checks. Run without EvalMath or a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
import torch

from research.dimensional_atlas.gym import tasks
from research.dimensional_atlas.gym.model import (LoopedMathTransformer,atlas_probe,
    set_base_trainable,set_adapter_trainable,set_sae_trainable)
from research.dimensional_atlas.gym.run import (model_digest,generate_replay,inspect_discovery,
    optimizer_steps,score,tensorize,run)
from argparse import Namespace


class TaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        torch.set_num_threads(1)

    def test_toy_oracle_and_query_budget(self):
        x=tasks.make_toy(15,case_id='x')
        yes=tasks.check_candidate(x,x.source_graph)
        self.assertTrue(yes['accepted'])
        found=tasks.verify_order(x,[tasks.PROGRAMS.index(x.source_graph)],budget=1)
        self.assertTrue(found['future_correct'])
        self.assertEqual(found['queries'],1)

    def test_public_never_contains_gold(self):
        x=tasks.make_toy(17,case_id='x')
        public=json.dumps(x.public)
        self.assertNotIn('source_graph',public)
        self.assertNotIn('future',public)
        self.assertEqual(tasks.encode(x).shape,(4,30))

    def test_signature_independent_of_id(self):
        a=tasks.make_toy(19,case_id='first')
        b=tasks.make_toy(19,case_id='second')
        self.assertEqual(a.signature,b.signature)

    def test_pin_refuses_fake_evalmath(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'oracle.py'
            file.write_text('pass\n')
            with self.assertRaisesRegex(ValueError,'source hash mismatch'):
                tasks.load_evalmath(file)

    def test_frontier_rejects_non_discovery(self):
        rows=[{'split':'test','case_id':'x','family':'iid','graph':['I','I','I'],
               'topology':'fork','confidence':0.5,'first_fault':9}]
        with self.assertRaisesRegex(ValueError,'discovery'):
            tasks.select_frontier(rows)

    def test_frontier_is_local_maximum(self):
        def row(case,fault,conf):
            return {'split':'discovery','case_id':case,'family':'iid',
                'graph':['I','N','R'],'topology':'fork','confidence':conf,'first_fault':fault}
        p=tasks.select_frontier([row('a',4,.6),row('b',8,.9),row('c',7,.7)])
        self.assertEqual([r['case_id'] for r in p],['b'])

    def test_fresh_generation_and_split(self):
        peak={'case_id':'x','family':'iid','topology':'fork','graph':['I','N','R'],
              'frontier_proxy':.5}
        public,gold=tasks.seal_new_evals([peak],per_peak=2)
        self.assertEqual(len(public),2)
        self.assertEqual(len(gold),2)
        self.assertNotIn('future',public[0])
        self.assertNotIn('source_graph',public[0])
        self.assertEqual(public[0]['public_sha256'],gold[0]['public_sha256'])
        with self.assertRaises(ValueError):
            tasks.seal_new_evals([peak],per_peak=2,forbidden={public[0]['public_sha256']})


class AtlasModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22)
        torch.set_num_threads(1)
        self.model=LoopedMathTransformer(hidden=16,heads=4,visits=2,sae_width=20,sae_k=3,vpd_rank=2)
        self.task=tasks.make_toy(12,case_id='case')

    def test_weight_tying_and_sparsity(self):
        x=tasks.encode(self.task).unsqueeze(0)
        out=self.model(x,use_vpd=False)
        self.assertEqual(len(out.states),2)
        self.assertEqual(tuple(out.logits.shape),(1,64))
        self.assertTrue(all((code>0).sum(-1).max()<=3 for code in out.codes))
        self.assertEqual(len(self.model.core.attn.in_proj_weight.shape),2)

    def test_adapter_shared_rank_and_zero_code(self):
        x=tasks.encode(self.task).unsqueeze(0)
        ref=self.model(x,use_vpd=False).logits
        control=self.model(x,use_vpd=True,code_override='zero').logits
        torch.testing.assert_close(ref,control,atol=1e-7,rtol=1e-7)
        self.assertEqual(self.model.vpd.read_factor.shape,(16,2))
        self.assertEqual(self.model.vpd.write_factor.shape,(2,16))

    def test_permutation_rejects_bad_visit_order(self):
        with self.assertRaisesRegex(ValueError,'permutation'):
            self.model(tasks.encode(self.task).unsqueeze(0),use_vpd=True,visit_order=(0,0))

    def test_stage_freezing(self):
        set_base_trainable(self.model,False)
        set_adapter_trainable(self.model,True)
        set_sae_trainable(self.model,False)
        self.assertFalse(any(p.requires_grad for p in self.model.core.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.sae.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.vpd.parameters()))

    def test_training_and_controls(self):
        train=[tasks.make_toy(300+i,case_id=str(i)) for i in range(12)]
        before=model_digest(self.model)
        history=optimizer_steps(self.model,train,stage='base',steps=2,batch=4,seed=1,lr=.001)
        self.assertEqual(history['steps'],2)
        self.assertNotEqual(before,model_digest(self.model))
        optimizer_steps(self.model,train,stage='sae',steps=2,batch=4,seed=1,lr=.001)
        optimizer_steps(self.model,train,stage='vpd',steps=2,batch=4,seed=1,lr=.001)
        scored=score(self.model,train,budget=3)
        self.assertEqual(scored['overall']['n'],12)
        self.assertLessEqual(scored['overall']['mean_exact_queries'],3)

    def test_atlas_probe_is_only_diagnostic(self):
        result=atlas_probe(self.model,self.task,eps=.01)
        self.assertEqual(result['status'],'finite_difference_diagnostic_only')
        self.assertEqual(len(result['visit_singular_values']),2)
        self.assertTrue(all(len(row)==2 for row in result['visit_singular_values']))


class RunTests(unittest.TestCase):
    def test_runner_small_smoke_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'receipt.json'
            args=Namespace(backend='toy',evalmath_source=None,output=path,seed=9,
                train_count=20,discovery_count=8,readout_count=8,steps=2,sae_steps=2,
                vpd_steps=2,replay_steps=2,batch=4,lr=.002,hidden=16,visits=2,
                sae_width=20,sae_k=3,vpd_rank=2,budget=4,peaks=2)
            r=run(args)
            self.assertEqual(r['status'],'executed_synthetic_gym')
            self.assertTrue(path.is_file())
            self.assertTrue(path.with_suffix('.gold.private.json').is_file())
            self.assertTrue(path.with_suffix('.generated.public.json').is_file())
            self.assertEqual(r['results']['zero_sae_codes']['future_accuracy'],r['results']['tied_base']['future_accuracy'])
            with self.assertRaises(FileExistsError):run(args)


if __name__=='__main__':unittest.main()
