"""RRC-12: reproducible graph split, direct-inference and finite-verifier audits."""
from __future__ import annotations
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
try:
    from . import graph11 as g,graph12 as b,model12,bench12,train12
except ImportError:
    import graph11 as g
    import graph12 as b
    import model12
    import bench12
    import train12
HERE=Path(__file__).resolve().parent

class Graph12Tests(unittest.TestCase):
    def test_semantic_alias_classes_and_disjoint_composition_split(self):
        self.assertEqual((len(b.CANONICAL),len(b.TRAIN_GRAPHS),len(b.TEST_GRAPHS)),(25,17,8))
        self.assertFalse(set(b.TRAIN_GRAPHS)&set(b.TEST_GRAPHS))
        self.assertTrue(all('S' in x and 'R' in x for x in b.TEST_GRAPHS))
        self.assertTrue(all(not ('S' in x and 'R' in x) for x in b.TRAIN_GRAPHS))
        rng=random.Random(4401)
        probes=[g.source(rng) for _ in range(10)]
        observed=[tuple(g.execute(f,graph) for f in probes) for graph in b.CANONICAL]
        self.assertEqual(len(set(observed)),len(b.CANONICAL))

    def test_full_dsl_in_training_without_test_compound(self):
        self.assertEqual(set(k for graph in b.TRAIN_GRAPHS for k in graph),set(g.OPS))
        # All graph positions have observed S/R labels where the syntax permits it.
        self.assertTrue(any(x[2]=='R' for x in b.TRAIN_GRAPHS))
        self.assertTrue(any(x[2]=='S' for x in b.TRAIN_GRAPHS))
        self.assertTrue(any('S' in x[:2] for x in b.TRAIN_GRAPHS))
        self.assertTrue(any('R' in x[:2] for x in b.TRAIN_GRAPHS))

    def test_generate_train_only_never_calls_test_source(self):
        with patch.object(b,'_product_source',side_effect=AssertionError('test source accessed')):
            d=b.dataset(include_test=False,train_count=20,dev_count=10)
        self.assertEqual(set(d),{'train','dev'})
        self.assertTrue(all(c.hidden_graph in b.TRAIN_GRAPHS for cases in d.values() for c in cases))

    def test_fresh_splits_and_collision_guards(self):
        d=b.dataset(train_count=40,dev_count=20)
        self.assertEqual({k:len(v) for k,v in d.items() if k not in ('train','dev')},b.TEST_COUNTS)
        train={c.visible for c in d['train']}
        self.assertTrue(all(c.visible not in train for s,cases in d.items() if s!='train' for c in cases))
        self.assertTrue(all(c.hidden_graph in b.TEST_GRAPHS for c in d['unseen_composition']+d['source_shift']))
        for x,y in zip(d['late_collision'][::2],d['late_collision'][1::2]):
            self.assertEqual(x.visible,y.visible)
            self.assertEqual(x.query,y.query)
            self.assertNotEqual(x.future,y.future)
            self.assertEqual(x.pair,y.pair)

    def test_exact_solver_and_out_of_grammar(self):
        d=b.dataset(train_count=30,dev_count=10)
        self.assertTrue(all(g.search(c)['graph'] is not None for c in d['unseen_composition']))
        self.assertTrue(all(g.search(c)['graph'] is None for c in d['out_of_grammar']))
        for c in d['out_of_grammar']:
            self.assertEqual(b.verify(c,g.search(c)['graph'])['accepted'],0)

    def test_demonstration_encoding_is_raw_only(self):
        c=b.dataset(include_test=False,train_count=10,dev_count=5)['train'][0]
        v=b.features(c)
        self.assertEqual(len(v),2*(g.DEGREE+1))
        self.assertEqual(len(v[0]),4)
        self.assertEqual(v[0][2:],(0.0,0.0))
        self.assertEqual(v[-1][2:],(1.0,1.0))
        self.assertTrue(all(all(abs(x)<4 for x in row) for row in v))

    def test_direct_neural_inference_does_not_execute_public_graphs(self):
        c=b.dataset(include_test=False,train_count=5,dev_count=3)['dev'][0]
        models,_=bench12.checkpoints()
        with patch.object(g,'execute',side_effect=AssertionError('interpreter leaked into model')):
            for kind in bench12.KINDS:
                proposed=model12.predict(c,models[kind])
                self.assertIn(proposed,g.GRAPHS)

    def test_direct_verifier_rejects_wrong_models(self):
        c=b.dataset(include_test=False,train_count=10,dev_count=5)['dev'][0]
        other=next(gr for gr in g.GRAPHS if g.execute(c.demos[0][0][:g.DEGREE+1],gr)!=c.demos[0][1][:g.DEGREE+1])
        self.assertEqual(b.verify(c,other)['accepted'],0)
        self.assertEqual(b.verify(c,g.search(c)['graph'])['accepted'],1)
        self.assertEqual(b.verify(c,None)['accepted'],0)

    def test_neural_predictions_invariant_to_future_mutations(self):
        d=b.dataset(train_count=10,dev_count=5)['late_collision']
        models,_=bench12.checkpoints()
        for a,z in zip(d[::2],d[1::2]):
            for kind in bench12.KINDS:
                self.assertEqual(model12.predict(a,models[kind]),model12.predict(z,models[kind]))
            p=g.search(a)['graph']
            self.assertEqual(p,g.search(z)['graph'])
            self.assertEqual(b.verify(a,p)['correct_future']+b.verify(z,p)['correct_future'],1)

    def test_source_and_checkpoint_locks(self):
        models,shas=bench12.checkpoints()
        self.assertEqual(set(models),set(bench12.KINDS))
        self.assertEqual(len(shas),3)
        self.assertEqual(models['recurrent']['parameter_count'],4044)
        self.assertEqual(models['feedforward']['parameter_count'],3990)
        self.assertEqual(models['shuffled_recurrent']['parameter_count'],4044)
        self.assertTrue(all(m['source_dsl_test_generation_during_training'] is False for m in models.values()))

    def test_parallel_torch_portable_parity_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable; CI uses pure Python inference')
        torch.set_num_threads(1)
        c=b.dataset(include_test=False,train_count=10,dev_count=5)['dev'][0]
        models,_=bench12.checkpoints()
        inp=train12.input_tensor([c])
        for kind,ck in models.items():
            net=train12.create_model(kind)
            tensors={n:torch.tensor(val) for n,val in ck['state_dict'].items()}
            net.load_state_dict(tensors)
            with torch.no_grad():live=net(inp)[0].tolist()
            standalone=model12.logits(c,ck)
            self.assertLess(max(abs(a-z) for a,z in zip(live,standalone)),2e-5,kind)

    def test_budget_and_recurrent_scans(self):
        models,_=bench12.checkpoints()
        self.assertEqual(model12.inference_mac(models['recurrent'],1),90240)
        self.assertEqual(model12.inference_mac(models['feedforward']),3944)
        self.assertGreater(model12.inference_mac(models['recurrent'],4),model12.inference_mac(models['recurrent'],1))
        with self.assertRaises(ValueError):model12.recurrent_logits([],models['recurrent'],scans=0)

    def test_frozen_source_bound_receipt(self):
        p=HERE/'results'/'rrc12_pilot.json';r=json.loads(p.read_text())
        self.assertEqual(r['source_sha256'],bench12.source_hashes())
        self.assertEqual(r['parameters']['test_counts'],b.TEST_COUNTS)
        self.assertEqual(r['scores']['unseen_composition']['exact_all_64']['correct_future'],160)
        self.assertEqual(r['scores']['unseen_composition']['recurrent_1']['correct_future'],0)
        self.assertEqual(r['scores']['unseen_composition']['untrained_preview_4']['correct_future'],159)
        self.assertEqual(r['scores']['source_shift']['exact_all_64']['correct_future'],64)
        self.assertEqual(r['scores']['out_of_grammar']['exact_all_64']['accepted'],0)
        self.assertEqual(r['scores']['late_collision']['exact_all_64']['false_accept'],8)
        self.assertTrue(all(v==0 for v in r['train_observed_prefix_aliases'].values()))

if __name__=='__main__':unittest.main()
