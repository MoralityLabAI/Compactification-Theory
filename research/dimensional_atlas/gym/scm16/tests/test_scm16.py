"""SCM-16 independent fiber algebra, local maxima and campaign regressions."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
import torch
from research.dimensional_atlas.gym.tasks import make_toy, PROGRAMS, encode
from research.dimensional_atlas.gym.run import optimizer_steps, model_digest
from research.dimensional_atlas.gym.scm16.fiber import FiberLoopedMathTransformer
from research.dimensional_atlas.gym.scm16.landscape import (neighbors,prefix_score,
                         local_maxima,mine_discovery,generate_fresh_challenges)
from research.dimensional_atlas.gym.scm16.campaign import (evaluate,paired_counts,
                         base_digest,run,fiber_mediation)
from research.dimensional_atlas.gym.scm16.verify_receipt import verify

class FiberStructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.model=FiberLoopedMathTransformer(hidden=16,heads=4,visits=3,
             sae_width=24,sae_k=3,vpd_rank=2,readout_dim=4)
        self.model.lock_fiber()
        self.task=make_toy(41,case_id='math:41')

    def test_frozen_readout_subspace_is_proper(self):
        q=self.model.vpd.readout_basis
        torch.testing.assert_close(q.T@q,torch.eye(4),rtol=1e-6,atol=1e-6)
        self.assertEqual(int(torch.linalg.matrix_rank(q)),4)
        self.assertFalse(self.model.head_norm.__class__.__name__.lower().startswith('layernorm'))
        with self.assertRaisesRegex(ValueError,'already frozen'):
            self.model.lock_fiber()

    def test_exact_readout_fiber_null(self):
        raw=torch.randn(2,4,16)
        parallel,perp=self.model.vpd.decompose(raw)
        q=self.model.vpd.readout_basis
        torch.testing.assert_close(perp@q,torch.zeros(2,4,4),rtol=0,atol=1e-5)
        torch.testing.assert_close(self.model.head[0].weight@perp[0,3],torch.zeros(4),rtol=0,atol=1e-5)
        torch.testing.assert_close(raw,parallel+perp,rtol=1e-6,atol=1e-6)

    def test_fiber_exact_last_visit_logit_null(self):
        h=torch.randn(1,4,16)
        code=torch.rand(1,4,24)
        _,_,visible,fiber=self.model.vpd(h,code,2)
        original=self.model.head(h[:,3]);changed=self.model.head(h[:,3]+fiber[:,3])
        torch.testing.assert_close(original,changed,rtol=0,atol=1e-6)
        self.assertTrue(torch.isfinite(visible).all())

    def test_shared_vpd_and_visit_shuffle(self):
        x=encode(self.task).unsqueeze(0)
        y=self.model(x,use_vpd=True,visit_order=(2,1,0))
        self.assertEqual(len(y.gates),3)
        self.assertEqual(self.model.vpd.read_factor.shape,(16,2))
        self.assertEqual(len(y.fiber_energies),3)

    def test_zero_code_equals_baseline(self):
        x=encode(self.task).unsqueeze(0)
        self.model.eval()
        a=self.model(x,use_vpd=False).logits
        b=self.model(x,use_vpd=True,code_override='zero').logits
        torch.testing.assert_close(a,b,rtol=0,atol=1e-6)

    def test_frozen_base_remains_after_sae_vpd_training(self):
        train=[make_toy(100+i,case_id=str(i)) for i in range(8)]
        original=base_digest(self.model)
        optimizer_steps(self.model,train,stage='sae',steps=2,batch=4,seed=4,lr=.001)
        optimizer_steps(self.model,train,stage='vpd',steps=2,batch=4,seed=5,lr=.001)
        self.assertEqual(original,base_digest(self.model))

    def test_intermediate_fiber_patch_and_mediation(self):
        result=fiber_mediation(self.model,self.task)
        self.assertEqual(len(result['by_visit']['fiber']),3)
        self.assertTrue(result['last_visit_fiber_is_readout_null'])
        self.assertGreaterEqual(result['by_visit']['visible'][-1],result['by_visit']['fiber'][-1])

    def test_bad_fiber_parameters_fail(self):
        with self.assertRaises(ValueError):
            FiberLoopedMathTransformer(hidden=16,readout_dim=16)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1,4,30),use_vpd=True,sector='any')
        with self.assertRaises(ValueError):
            self.model(torch.randn(1,4,30),use_vpd=True,visit_order=(0,0,2))

class LandscapeTests(unittest.TestCase):
    def test_graph_neighbors_are_nine(self):
        graph=('I','S','R')
        self.assertEqual(len(set(neighbors(graph))),9)
        self.assertNotIn(graph,list(neighbors(graph)))

    def test_exact_local_maximum_criterion(self):
        task=make_toy(23,case_id='discovery:23')
        all_values={g:prefix_score(task,g) for g in PROGRAMS}
        for item in local_maxima(task):
            graph=tuple(item['graph']);value=item['prefix_score']
            self.assertTrue(all(value>=all_values[n] for n in neighbors(graph)))
            self.assertTrue(any(value>all_values[n] for n in neighbors(graph)))

    def test_discovery_provenance_and_queries(self):
        tasks=[make_toy(500+i,case_id='discovery:'+str(i)) for i in range(8)]
        r=mine_discovery(tasks,max_tasks=6,peaks=3)
        self.assertEqual(r['verifier_queries'],384)
        self.assertLessEqual(r['peak_count'],3)
        self.assertGreater(r['peak_count'],0)
        bad=[make_toy(2,case_id='readout:wrong')]
        with self.assertRaisesRegex(ValueError,'forbidden'):
            mine_discovery(bad)

    def test_challenge_public_signature_disjoint(self):
        tasks=[make_toy(700+i,case_id='discovery:'+str(i)) for i in range(8)]
        peaks=mine_discovery(tasks,max_tasks=8,peaks=2)['peaks']
        challenges=generate_fresh_challenges(peaks,backend='toy',per_peak=2,
                      forbidden={t.signature for t in tasks})
        self.assertEqual(len(challenges),4)
        self.assertEqual(len({t.signature for t in challenges}),4)
        self.assertTrue(all('future' not in task.public for task in challenges))

class GymIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.set_num_threads(1)

    def test_exact_verifier_and_matched_records(self):
        model=FiberLoopedMathTransformer(hidden=16,heads=4,visits=2,
                         sae_width=20,sae_k=3,vpd_rank=2,readout_dim=4)
        model.lock_fiber()
        tasks=[make_toy(1000+i,case_id='r:'+str(i)) for i in range(6)]
        a=evaluate(model,tasks,budget=3,policy='tied')
        b=evaluate(model,tasks,budget=3,policy='joint')
        self.assertEqual(a['overall']['n'],6)
        self.assertLessEqual(a['overall']['mean_queries'],3)
        self.assertEqual(sum(paired_counts(a,b).values()),6)
        c=evaluate(model,tasks,budget=3,policy='zero_codes')
        self.assertEqual(a['records'],c['records'])

    def test_committed_public_receipt_integrity(self):
        root=Path(__file__).resolve().parents[1] / 'results'
        r=verify(root/'scm16_toy_3seed_receipt.json',root/'scm16_toy_generated_public.json')
        self.assertEqual(r['seed_count'],3)

    def test_minimal_two_seed_full_run_and_protected_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'scm16.json'
            args=Namespace(backend='toy',evalmath_source=None,output=output,
                seed=18,seed_count=2,train_count=22,discovery_count=9,
                readout_count=8,mining_tasks=5,peaks=2,replay_per_peak=2,
                challenge_per_peak=2,base_steps=2,sae_steps=2,vpd_steps=2,
                replay_steps=2,batch=4,lr=.002,hidden=16,heads=4,
                readout_dim=4,visits=2,sae_width=20,sae_k=3,
                vpd_rank=2,budget=4)
            result=run(args)
            self.assertEqual(len(result['results']),2)
            self.assertEqual(result['readout_count'],8)
            self.assertEqual(result['mining']['verifier_queries'],5*64)
            self.assertTrue(output.with_suffix('.generated.public.json').is_file())
            self.assertTrue(output.with_suffix('.gold.private.json').is_file())
            public=output.with_suffix('.generated.public.json').read_text()
            self.assertNotIn('source_program',public)
            self.assertNotIn('"future"',public)
            self.assertTrue(all(result['results'][i]['official']['zero_codes']['overall'] ==
                                result['results'][i]['official']['tied']['overall'] for i in (0,1)))
            with self.assertRaises(FileExistsError):run(args)

if __name__=='__main__':unittest.main()
