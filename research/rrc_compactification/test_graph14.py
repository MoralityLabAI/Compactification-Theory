"""RRC-14 exact, reproducible, budgeted reversible graph-edit tests."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from . import graph11 as g, graph12 as b12, graph14 as b, policy14 as p, train14 as t, bench14 as bench
except ImportError:
    import graph11 as g
    import graph12 as b12
    import graph14 as b
    import policy14 as p
    import train14 as t
    import bench14 as bench

HERE=Path(__file__).resolve().parent


class Graph14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=b.dataset(include_test=False,train_count=12,dev_count=5)
        cls.models,cls.digests,cls.prior=bench.checkpoints()
        cls.sample=cls.train['dev'][0]

    def test_reversible_single_token_graph_edges(self):
        for parent in g.GRAPHS:
            neighbors=b.neighborhood(parent)
            self.assertEqual(len(neighbors),9)
            self.assertEqual(len({a for a,_,_ in neighbors}),9)
            for child,position,operation in neighbors:
                self.assertTrue(b.reverse_edge(parent,child))
                self.assertEqual(child[position],operation)
                self.assertIn(parent,[x for x,_,_ in b.neighborhood(child)])
        self.assertFalse(b.reverse_edge(('I','I','I'),('S','R','I')))

    def test_independent_coefficient_arithmetic(self):
        f=(1,2,1,0,0,0)
        self.assertEqual(g.mul(f,f),(1,4,6,4,1,0))
        self.assertEqual(g.mul(f,g.inv(f)),(1,0,0,0,0,0))
        self.assertNotEqual(g.unary(g.unary(f,'S'),'N'),g.unary(g.unary(f,'N'),'S'))

    def test_fresh_source_roster_no_test_labels_during_fit(self):
        with patch.object(b,'TEST_GRAPHS',side_effect=AssertionError('test requested')):
            x=b.dataset(include_test=False,train_count=16,dev_count=7)
        self.assertEqual(set(x),{'train','dev'})
        self.assertTrue(all(c.hidden_graph in b.TRAIN_GRAPHS for c in x['train']+x['dev']))
        whole=b.dataset(train_count=15,dev_count=7)
        self.assertEqual({k:len(v) for k,v in whole.items() if k not in ('train','dev')},b.COUNTS)
        self.assertTrue(all(c.hidden_graph in b.TEST_GRAPHS for name in ('unseen_composition','source_shift') for c in whole[name]))
        known={c.visible for c in whole['train']}
        self.assertTrue(all(c.visible not in known for name,group in whole.items() if name!='train' for c in group))

    def test_action_features_public_only(self):
        c=self.sample;graph=('I','I','I')
        state=b.feedback(c,graph)
        feature=b.action_features(c,graph,state)
        self.assertEqual(len(feature),p.FEATURES)
        sentinel=SimpleNamespace(identifier=c.identifier,demos=c.demos,visible=c.visible)
        self.assertEqual(state,b.feedback(sentinel,graph))
        self.assertEqual(feature,b.action_features(sentinel,graph,b.feedback(sentinel,graph)))
        self.assertEqual(len(b.graph_onehot(graph)),12)
        self.assertEqual(len(state['residual']),16)

    def test_teacher_uses_counterfactuals_only_for_train(self):
        with patch.object(b,'feedback',wraps=b.feedback) as spy:
            label,new=b.teacher(self.sample,('I','I','I'))
        self.assertEqual(spy.call_count,9)
        self.assertEqual(new,b.neighborhood(('I','I','I'))[label][0])

    def test_model_parameters_and_source_checkpoints(self):
        self.assertEqual(self.models['recurrent']['parameters'],7353)
        self.assertEqual(self.models['feedforward']['parameters'],7393)
        self.assertEqual(self.models['shuffled']['parameters'],7353)
        self.assertLess(abs(7353-7393)/7353,0.01)
        self.assertEqual(self.models['recurrent']['source_hashes'],t.source_hashes())
        self.assertEqual(self.models['recurrent']['train_roster_sha256'],b.digest(b.dataset(include_test=False)['train']))
        self.assertFalse(self.models['recurrent']['test_generated_during_training'])
        self.assertEqual(self.models['recurrent']['teacher_future_access'],False)

    def test_exact_feedback_consumes_one_two_demo_query(self):
        c=self.sample;graph=('I','I','I')
        with patch.object(g,'execute',wraps=g.execute) as spy:
            b.feedback(c,graph)
        self.assertEqual(spy.call_count,2)
        self.assertEqual(b.cost(1),{'graph_queries':1,'demo_executions':2,
                                     'primitive_calls':8,'integer_loop_proxy':1014})

    def test_policy_only_reads_visible_input(self):
        sentinel=SimpleNamespace(identifier=self.sample.identifier,demos=self.sample.demos,visible=self.sample.visible)
        for arm in ('recurrent','feedforward','residual_frontier'):
            candidate=bench.edits(sentinel,arm,self.models,self.prior,budget=3)
            self.assertLessEqual(candidate['queries'],3)
            self.assertEqual(candidate['queries'],candidate['unique_graphs'])

    def test_exact_budget_and_cache_one_call_each(self):
        c=self.sample
        for arm in ('recurrent','recurrent_no_history','recurrent_blind_residual',
                    'feedforward','shuffled','lexical_frontier','residual_frontier','random_frontier'):
            with patch.object(b,'feedback',wraps=b.feedback) as spy:
                result=bench.edits(c,arm,self.models,self.prior,budget=9)
            self.assertEqual(spy.call_count,result['queries'])
            self.assertEqual(len(set(result['visited'])),result['queries'])
            self.assertLessEqual(result['queries'],9)
            if result['graph'] is not None:
                self.assertEqual(bench.grade(c,result)['accepted'],1)

    def test_zero_feedback_and_no_history_ablation(self):
        c=self.sample
        for arm in ('recurrent','recurrent_no_history','recurrent_blind_residual'):
            result=bench.edits(c,arm,self.models,self.prior,budget=4)
            self.assertLessEqual(result['queries'],4)
        with self.assertRaises(ValueError):bench.edits(c,'recurrent',self.models,self.prior,budget=0)
        with self.assertRaises(ValueError):bench.edits(c,'recurrent',self.models,self.prior,budget=65)

    def test_deterministic_future_collision_seed(self):
        a,c=b.dataset(train_count=12,dev_count=5)['late_collision'][:2]
        self.assertEqual(a.visible,c.visible)
        self.assertEqual(a.query,c.query)
        self.assertNotEqual(a.future,c.future)
        for arm in ('recurrent','feedforward','residual_frontier','random_frontier'):
            x=bench.edits(a,arm,self.models,self.prior,budget=12)
            y=bench.edits(c,arm,self.models,self.prior,budget=12)
            self.assertEqual(x['graph'],y['graph'])
            self.assertEqual(x['visited'],y['visited'])
            if x['graph'] is not None:
                self.assertEqual(b.verify_future(a,x['graph'])['future_correct']+
                                 b.verify_future(c,y['graph'])['future_correct'],1)

    def test_inconsistent_demos_rejected_by_complete_exact_search(self):
        cases=b.dataset(train_count=12,dev_count=5)['out_of_grammar']
        self.assertEqual(len(cases),20)
        self.assertTrue(all(g.search(c)['graph'] is None for c in cases))
        self.assertTrue(all(bench.fixed(c,'exact_search')['graph'] is None for c in cases))

    def test_portable_pytorch_parity(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch optional for inference parity')
        import torch.nn as nn
        torch.set_num_threads(1)
        graph=('I','I','I');signal=b.feedback(self.sample,graph)
        x=list(b.action_features(self.sample,graph,signal))
        vector=torch.tensor([x],dtype=torch.float32)
        for kind in ('recurrent','feedforward'):
            checkpoint=self.models[kind]
            s=checkpoint['state_dict']
            if kind=='recurrent':
                cell=nn.GRUCell(p.FEATURES,p.HIDDEN)
                head=nn.Linear(p.HIDDEN,p.ACTIONS)
                cell.load_state_dict({key[len('cell.'):]:torch.tensor(value)
                                      for key,value in s.items() if key.startswith('cell.')})
                head.load_state_dict({key[len('head.'):]:torch.tensor(value)
                                      for key,value in s.items() if key.startswith('head.')})
                h=torch.zeros((1,p.HIDDEN))
                portable=None
                for _ in range(3):
                    with torch.no_grad():
                        h=cell(vector,h);reference=head(h)[0].tolist()
                    got,portable=p.step(x,checkpoint,portable)
                    self.assertLess(max(abs(u-v) for u,v in zip(got,reference)),2e-5)
            else:
                fc1=nn.Linear(p.FEATURES,p.FF_HIDDEN);head=nn.Linear(p.FF_HIDDEN,p.ACTIONS)
                fc1.load_state_dict({k[len('fc1.'):]:torch.tensor(v) for k,v in s.items() if k.startswith('fc1.')})
                head.load_state_dict({k[len('head.'):]:torch.tensor(v) for k,v in s.items() if k.startswith('head.')})
                with torch.no_grad():reference=head(torch.tanh(fc1(vector)))[0].tolist()
                got,_=p.step(x,checkpoint)
                self.assertLess(max(abs(u-v) for u,v in zip(got,reference)),2e-5)

    def test_frozen_source_bound_receipt(self):
        obj=json.loads((HERE/'results'/'rrc14_pilot.json').read_text())
        self.assertEqual(obj['source_sha256'],bench.hashes())
        self.assertEqual(obj['checkpoint_sha256'],self.digests)
        self.assertEqual(obj['parameters']['edit_budget_graph_queries'],12)
        self.assertEqual(obj['metrics']['unseen_composition']['recurrent']['accepted'],53)
        self.assertEqual(obj['metrics']['unseen_composition']['feedforward']['accepted'],63)
        self.assertEqual(obj['metrics']['unseen_composition']['frozen_initializer']['accepted'],0)
        self.assertEqual(obj['metrics']['unseen_composition']['residual_frontier']['accepted'],78)
        self.assertEqual(obj['metrics']['source_shift']['recurrent']['accepted'],16)
        self.assertEqual(obj['metrics']['out_of_grammar']['exact_search']['accepted'],0)
        self.assertEqual(obj['metrics']['late_collision']['exact_search']['false_accept'],8)

    def test_source_and_graph_invariance_of_policy_syntax(self):
        graph=('R','S','I')
        c=self.sample
        r=b.feedback(c,graph)
        x=b.action_features(c,graph,r)
        self.assertEqual(x[-16:],r['residual'])
        for op in b.OPS:
            self.assertEqual(sum(int(z==op) for z in graph),sum(int(graph[i]==op) for i in range(3)))


if __name__=='__main__':unittest.main()
