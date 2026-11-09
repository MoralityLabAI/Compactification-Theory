"""RRC-13 graph construction: algebra, leakage, feedback and replay checks."""
from __future__ import annotations
from dataclasses import replace
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from . import graph11 as g, graph12 as b12, graph13 as b, policy13 as p, train13 as t, bench13
except ImportError:
    import graph11 as g
    import graph12 as b12
    import graph13 as b
    import policy13 as p
    import train13 as t
    import bench13

HERE=Path(__file__).resolve().parent

class AlgebraAndRoster(unittest.TestCase):
    def test_new_source_family_and_exact_formal_series(self):
        rng=random.Random(102)
        for _ in range(20):
            f=b._product_source(rng)
            self.assertEqual(f[0],1)
            self.assertEqual(g.mul(f,g.inv(f)),(1,)+(0,)*g.HORIZON)
            self.assertEqual(g.unary(g.unary(f,'R'),'R'),f)

    def test_semantic_composition_split(self):
        self.assertEqual((len(b12.CANONICAL),len(b.TRAIN_GRAPHS),len(b.TEST_GRAPHS)),(25,17,8))
        self.assertFalse(set(b.TRAIN_GRAPHS)&set(b.TEST_GRAPHS))
        self.assertEqual(set(v for graph in b.TRAIN_GRAPHS for v in graph),set(g.OPS))
        self.assertTrue(all('S' in gr and 'R' in gr for gr in b.TEST_GRAPHS))
        rng=random.Random(232)
        extra=[g.source(rng) for _ in range(7)]
        sigs=[tuple(g.execute(x,gr) for x in extra) for gr in b12.CANONICAL]
        self.assertEqual(len(set(sigs)),25)

    def test_training_fully_excludes_test_generator(self):
        with patch.object(b,'_product_source',side_effect=AssertionError('test-only source called')):
            d=b.dataset(include_test=False,train_count=16,dev_count=8)
        self.assertEqual(set(d),{'train','dev'})
        self.assertTrue(all(c.hidden_graph in b.TRAIN_GRAPHS for group in d.values() for c in group))

    def test_full_split_sizes_and_disjoint_visible(self):
        d=b.dataset(train_count=22,dev_count=10)
        self.assertEqual({k:len(v) for k,v in d.items() if k not in ('train','dev')},b.TEST_COUNTS)
        known={c.visible for c in d['train']}
        self.assertTrue(all(c.visible not in known for split,cases in d.items()
                            if split!='train' for c in cases))
        self.assertTrue(all(c.hidden_graph in b.TEST_GRAPHS for group in
                            ('unseen_composition','source_shift') for c in d[group]))

    def test_negatives_and_collision_verification(self):
        d=b.dataset(train_count=20,dev_count=8)
        self.assertTrue(all(g.search(c)['graph'] is None for c in d['out_of_grammar']))
        for a,z in zip(d['late_collision'][::2],d['late_collision'][1::2]):
            self.assertEqual(a.visible,z.visible)
            self.assertEqual(a.query,z.query)
            self.assertNotEqual(a.future,z.future)
            found=g.search(a)['graph']
            self.assertEqual(found,g.search(z)['graph'])
            self.assertEqual(b.certificate(a,found)['correct_future']+
                             b.certificate(z,found)['correct_future'],1)

class FeedbackAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,_=bench13.checkpoints()
        cls.example=b.dataset(include_test=False,train_count=10,dev_count=5)['dev'][0]

    def test_feedback_one_selected_partial_per_demo(self):
        c=self.example
        original=g.execute;observed=[]
        def tracked(f,graph):
            observed.append(graph)
            return original(f,graph)
        with patch.object(g,'execute',side_effect=tracked):
            a=b.residual_feedback(c,'N')
            ab=b.residual_feedback(c,'N','R')
        self.assertEqual(len(a),16);self.assertEqual(len(ab),16)
        self.assertEqual(observed,[('N','I','I')]*2+[('N','R','I')]*2)
        self.assertTrue(all(-3.1<x<4 for x in a+ab))

    def test_exact_feedback_against_matching_graph(self):
        src=(1,2,0,1)+(0,)*(g.HORIZON-3)
        y=g.execute(src,('I','I','I'))
        c=b.Case('t','t',((src,y),(src,y)),src,y,('I','I','I'))
        fb=b.residual_feedback(c,'I','I')
        self.assertEqual(fb[:3],(1.0,1.0,0.0))
        self.assertEqual(fb[8:11],(1.0,1.0,0.0))
        self.assertEqual(fb[3:8]+fb[11:16],(0.,)*10)

    def test_full_policy_reads_no_hidden_future(self):
        c=self.example
        other=replace(c,hidden_graph=None,future=(1,)+(777,)*g.HORIZON)
        for arm,flag in [('active',True),('active',False),('passive',False)]:
            ck=self.models[arm]
            self.assertEqual(p.infer(c,ck,feedback_enabled=flag),p.infer(other,ck,feedback_enabled=flag))

    def test_active_exactly_two_partial_queries_passive_zero(self):
        c=self.example;original=g.execute
        for arm,expect in [('active',4),('passive',0)]:
            tracker=[]
            def observed(f,graph):
                tracker.append(graph);return original(f,graph)
            with patch.object(g,'execute',side_effect=observed):
                r=p.infer(c,self.models[arm])
            self.assertEqual(len(tracker),expect)
            self.assertEqual(r['partial_queries'],expect//2)
            self.assertTrue(all(tuple(v) in g.GRAPHS for v in tracker))
        with patch.object(g,'execute',side_effect=AssertionError('feedback disabled')):
            self.assertEqual(p.infer(c,self.models['active'],feedback_enabled=False)['partial_queries'],0)

    def test_teacher_arrays_depend_on_observed_demos_only(self):
        try:import torch
        except ImportError:self.skipTest('optional PyTorch for training')
        c=self.example
        altered=replace(c,future=(1,)+(14,)*g.HORIZON)
        a=t.arrays([c]);z=t.arrays([altered])
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,z)))

    def test_no_catalog_search_in_active_inference(self):
        with patch.object(g,'search',side_effect=AssertionError('catalog search leakage')):
            with patch.object(g,'guided',side_effect=AssertionError('preview search leakage')):
                r=p.infer(self.example,self.models['active'])
        self.assertEqual(r['partial_queries'],2)
        self.assertEqual(len(r['graph']),3)

    def test_parameter_match_and_checkpoint_locks(self):
        self.assertEqual(self.models['active']['parameter_count'],6148)
        self.assertEqual(self.models['passive']['parameter_count'],6148)
        self.assertEqual(self.models['active']['source_hashes'],t.hashes())
        self.assertTrue(self.models['active']['training_teacher_labels'])
        self.assertEqual(self.models['active']['partial_calls_train_per_case'],20)
        self.assertEqual(self.models['passive']['partial_calls_train_per_case'],0)
        self.assertEqual(p.inference_macs(self.models['active']),12936)
        self.assertEqual(p.inference_macs(self.models['passive']),12936)

    def test_torch_vs_portable_teacher_path(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:self.skipTest('PyTorch optional')
        torch.set_num_threads(1)
        c=self.example
        for arm in ('active','passive'):
            ck=self.models[arm];net=t.create_model()
            net.load_state_dict({k:torch.tensor(v) for k,v in ck['state_dict'].items()});net.eval()
            raw=torch.tensor([b.raw_features(c)],dtype=torch.float32)
            h=torch.tanh(net.encoder(raw))
            prev=torch.zeros((1,4));fb=torch.zeros((1,b.FEEDBACK_SIZE));live=[]
            for step in range(3):
                onehot=F.one_hot(torch.tensor([step]),3).float()
                h=net.cell(torch.cat([prev,fb,onehot],dim=1),h)
                live.append(net.head(h)[0].detach().tolist())
                op=c.hidden_graph[step]
                prev=F.one_hot(torch.tensor([b.OPS.index(op)]),4).float()
                if step<2 and arm=='active':
                    fb=torch.tensor([b.residual_feedback(c,c.hidden_graph[0],
                        c.hidden_graph[1] if step==1 else None)])
                else:fb=torch.zeros_like(fb)
            portable=p.infer(c,ck,forced_prefix=c.hidden_graph)['logits']
            self.assertLess(max(abs(a-z) for row1,row2 in zip(live,portable)
                                for a,z in zip(row1,row2)),3e-5)

class LedgersAndReplay(unittest.TestCase):
    def test_cost_accounting_and_greedy_budget(self):
        self.assertEqual(b.policy_work(2,1),{'partial_queries':2,'full_verifications':1,
            'primitive_calls':24,'integer_loop_proxy':3042})
        self.assertEqual(b.policy_work(12,1)['integer_loop_proxy'],13182)
        c=b.dataset(include_test=False,train_count=10,dev_count=5)['dev'][0]
        orig=g.execute;count=0
        def tracker(*args):
            nonlocal count
            count+=1;return orig(*args)
        with patch.object(g,'execute',side_effect=tracker):proposed=b.greedy_construct(c)
        self.assertEqual(count,24)
        self.assertIn(proposed,g.GRAPHS)
        with self.assertRaises(ValueError):b.residual_feedback(c,'X')
        with self.assertRaises(ValueError):p.infer(c,bench13.checkpoints()[0]['active'],forced_prefix=('X',))

    def test_one_verifier_and_true_negative_boundary(self):
        c=b.dataset(train_count=10,dev_count=5)['out_of_grammar'][0]
        models,_=bench13.checkpoints()
        for arm in ('active','passive','greedy_stagewise','frozen_rrc12'):
            graph,ledger,_=bench13.candidate(c,arm,models)
            self.assertEqual(ledger['full_verifications'],1)
            self.assertEqual(b.certificate(c,graph)['accepted'],0)

    def test_source_bound_frozen_receipt(self):
        record=json.loads((HERE/'results/rrc13_pilot.json').read_text())
        self.assertEqual(record['source_sha256'],bench13.sources())
        self.assertEqual(record['parameters']['test_counts'],b.TEST_COUNTS)
        self.assertEqual(record['metrics']['familiar']['active']['correct_future'],25)
        self.assertEqual(record['metrics']['familiar']['passive']['correct_future'],12)
        self.assertEqual(record['metrics']['familiar']['active_no_feedback']['correct_future'],8)
        self.assertEqual(record['metrics']['unseen_composition']['active']['correct_future'],0)
        self.assertEqual(record['metrics']['unseen_composition']['exact_search']['correct_future'],120)
        self.assertEqual(record['metrics']['source_shift']['exact_search']['correct_future'],48)
        self.assertEqual(record['metrics']['source_shift']['preview_4']['correct_future'],45)
        self.assertEqual(record['metrics']['out_of_grammar']['exact_search']['accepted'],0)
        self.assertEqual(record['metrics']['late_collision']['exact_search']['false_accept'],8)
        self.assertTrue(all(v==0 for v in record['train_prefix_aliases'].values()))

if __name__=='__main__':unittest.main()
