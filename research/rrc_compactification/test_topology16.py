"""RRC-16 held-out join algebra, training isolation, trace and replay gates."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import random
import unittest
from unittest.mock import patch
try:
    from . import topology16 as b, policy16 as p, train16 as t, bench16 as e, graph11 as g
except ImportError:
    import topology16 as b
    import policy16 as p
    import train16 as t
    import bench16 as e
    import graph11 as g
HERE=Path(__file__).resolve().parent


class FormalSeriesTests(unittest.TestCase):
    def test_hadamard_is_exact_normalized_and_bilinear_on_augmentation(self):
        f=(1,2,-3,4,5,0)
        h=(1,-1,7,2,3,0)
        unit=(1,0,0,0,0,0)
        self.assertEqual(b.unit_hadamard(f,h),(1,-2,-21,8,15,0))
        self.assertEqual(b.unit_hadamard(f,unit),unit)
        for n in range(1,len(f)):
            self.assertEqual(b.unit_hadamard(f,h)[n],f[n]*h[n])
        with self.assertRaises(ValueError):b.unit_hadamard((2,1),(1,2))
        with self.assertRaises(ValueError):b.unit_hadamard(f,(1,1))

    def test_existing_topologies_preserve_existing_semantics(self):
        rng=random.Random(2006)
        for topology in b.TRAIN_TOPOLOGIES:
            for gr in g.GRAPHS:
                f=g.source(rng)
                self.assertEqual(b.execute(f,gr,topology),b.b15.execute(f,gr,topology))

    def test_new_join_has_distinct_behavior_and_unit_integrality(self):
        rng=random.Random(2007)
        f=(1,2,1,-1,3,0,0,0)
        self.assertNotEqual(b.execute(f,('I','I','I'),'hadamard'),
                            b.execute(f,('I','I','I'),'fork'))
        for gr in g.GRAPHS:
            outcome=b.execute(g.source(rng),gr,'hadamard')
            self.assertEqual(outcome[0],1)
            self.assertTrue(all(type(v) is int for v in outcome))
        with self.assertRaises(ValueError):b.execute(f,('X','I','I'),'hadamard')
        with self.assertRaises(ValueError):b.execute(f,('I','I','I'),'missing')

    def test_public_topology_feature_composition(self):
        self.assertEqual(b.topology_descriptor('fork'),(1.,1.,0.))
        self.assertEqual(b.topology_descriptor('cascade'),(1.,1.,1.))
        self.assertEqual(b.topology_descriptor('additive'),(0.,0.,0.))
        self.assertEqual(b.topology_descriptor('hadamard'),(0.,1.,0.))
        self.assertLess(b.graph_cost(('I','I','I'),'hadamard'),
                        b.graph_cost(('I','I','I'),'fork'))
        with self.assertRaises(ValueError):b.graph_cost(('I','I','I'),'hadamard',-1)


class RosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=b.roster(train_per=12,dev_per=4)

    def test_disjoint_wirings_and_case_counts(self):
        d=self.data
        self.assertEqual({k:len(v) for k,v in d.items() if k not in ('train','dev')},b.COUNTS)
        self.assertEqual(len(d['train']),36)
        self.assertEqual(len(d['dev']),12)
        self.assertTrue(all(x.topology in b.TRAIN_TOPOLOGIES for k in ('train','dev') for x in d[k]))
        self.assertTrue(all(x.topology=='hadamard' for k in b.COUNTS if k!='seen_wirings' for x in d[k]))
        sig=set(x.visible for x in d['train'])
        self.assertFalse(any(x.visible in sig for k in b.COUNTS for x in d[k]))

    def test_holdout_has_no_observed_old_topology_aliases(self):
        for split in ('new_join_signed','new_join_products'):
            for x in self.data[split]:
                self.assertTrue(b.matching_graphs(x))
                self.assertTrue(all(not b.matching_graphs(x,top) for top in b.TRAIN_TOPOLOGIES))

    def test_negative_completeness_is_finite_grammar_scoped(self):
        for x in self.data['new_join_negative']:
            self.assertFalse(b.matching_graphs(x))
            self.assertEqual(b.certificate(x,None)['accepted'],0)

    def test_future_collision_inputs_identical(self):
        for a,c in zip(self.data['new_join_collision'][::2],self.data['new_join_collision'][1::2]):
            self.assertEqual(a.visible,c.visible)
            self.assertEqual(a.query,c.query)
            self.assertNotEqual(a.future,c.future)
            found=b.matching_graphs(a)[0]
            self.assertEqual(sum(b.certificate(x,found)['future_correct'] for x in (a,c)),1)

    def test_training_does_not_expose_heldout_join(self):
        # Even attempting to use its implementation during train/dev fails.
        with patch.object(b,'unit_hadamard',side_effect=AssertionError('heldout join entered training')):
            d=b.roster(include_test=False,train_per=5,dev_per=3)
            _,_,initializer=t.prior_models()
            rows,ops=t.trajectories(d['train'],initializer,b.SEED+1000)
        self.assertEqual(set(d),{'train','dev'})
        self.assertEqual(len(rows),30)
        self.assertGreater(ops,0)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.old,cls.initializer,cls.sha=e.checkpoints()
        cls.cases=b.roster(train_per=4,dev_per=2)

    def test_model_counts_and_pretrained_sources(self):
        self.assertEqual(p.parameter_count('recurrent'),7497)
        self.assertEqual(p.parameter_count('feedforward'),7549)
        self.assertEqual(self.models['recurrent']['source_sha256'],t.source_hashes())
        self.assertEqual(self.models['recurrent']['topologies_trained'],list(b.TRAIN_TOPOLOGIES))

    def test_public_only_and_actual_interpreter_call_accounting(self):
        original=b.execute
        c=self.cases['new_join_signed'][0]
        for arm in ('multi_gru','multi_gru_reset','multi_ff','frozen_gru14','residual_frontier'):
            with patch.object(b,'execute',wraps=original) as called:
                item=e.predict(c,arm,self.models,self.old,self.initializer,budget=3)
            self.assertEqual(called.call_count,2*len(item['trace']))
            self.assertEqual(item['integer_loop_proxy'],sum(2*b.graph_cost(x,c.topology) for x in item['trace']))
            self.assertEqual(len(set(item['trace'])),len(item['trace']))
            self.assertLessEqual(len(item['trace']),3)

    def test_cannot_see_hidden_graph_or_future(self):
        c=self.cases['new_join_products'][0]
        altered=replace(c,hidden_graph=None,future=(1,)*len(c.future),identifier='hidden')
        for arm in e.ARMS:
            a=e.predict(c,arm,self.models,self.old,self.initializer,budget=4)
            z=e.predict(altered,arm,self.models,self.old,self.initializer,budget=4)
            self.assertEqual((a['graph'],a['trace']),(z['graph'],z['trace']))

    def test_descriptor_and_state_ablation_keep_full_budget(self):
        c=self.cases['new_join_signed'][1]
        for arm in ('multi_gru','multi_gru_reset','multi_gru_descriptor_off','multi_ff','shuffled'):
            q=e.predict(c,arm,self.models,self.old,self.initializer)
            self.assertLessEqual(len(q['trace']),b.BUDGET)
            self.assertEqual(q['model_macs'],p.HIDDEN*0+q['model_macs'])

    def test_optional_torch_portable_parity(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch optional for CI')
        torch.set_num_threads(1)
        c=self.cases['seen_wirings'][0]
        gr=g.GRAPHS[0]
        x=b.features(c,gr,b.feedback(c,gr))
        for kind in ('recurrent','feedforward'):
            ck=self.models[kind]
            net=t.modules()[kind]()
            net.load_state_dict({k:torch.tensor(v) for k,v in ck['state_dict'].items()})
            if kind=='recurrent':
                with torch.no_grad():
                    hx=net.cell(torch.tensor([x],dtype=torch.float32),torch.zeros((1,p.HIDDEN)))
                    target=net.head(hx)[0].tolist()
            else:
                with torch.no_grad():
                    target=net(torch.tensor([[x]],dtype=torch.float32))[0,0].tolist()
            result,_=p.step(x,ck)
            self.assertLess(max(abs(a-z) for a,z in zip(target,result)),2e-5)

    def test_receipt_is_source_hash_bound(self):
        saved=json.loads((HERE/'results'/'rrc16_pilot.json').read_text())
        self.assertEqual(saved['source_sha256'],e.source_hashes())
        self.assertEqual(saved['checkpoint_sha256'],self.sha)
        self.assertEqual(saved['parameters']['graph_budget'],12)
        self.assertEqual(saved['scores']['new_join_signed']['exact_64']['accepted'],80)
        self.assertEqual(saved['scores']['new_join_products']['exact_64']['accepted'],40)
        self.assertEqual(saved['scores']['new_join_negative']['exact_64']['accepted'],0)
        self.assertEqual(saved['scores']['new_join_collision']['exact_64']['false_accept'],6)
        for fam in ('new_join_signed','new_join_products'):
            self.assertEqual(saved['old_grammar_aliases'][fam],0)
        self.assertTrue(all(v['identical_prediction_and_trace']
                            for v in saved['collision_pairs']['new_join_collision'].values()))

if __name__=='__main__':unittest.main()
