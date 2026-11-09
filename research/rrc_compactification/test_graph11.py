"""Independent algebra, data boundaries, graph induction and replay tests."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    from . import graph11 as g,train11 as t,bench11 as b
except ImportError:
    import graph11 as g
    import train11 as t
    import bench11 as b
HERE=Path(__file__).resolve().parent

class Graph11Tests(unittest.TestCase):
    def test_inverse_and_square_exact(self):
        f=(1,2,1,0,0,0)
        self.assertEqual(g.inv(f),(1,-2,3,-4,5,-6))
        self.assertEqual(g.mul((1,1,0,0),(1,1,0,0)),(1,2,1,0))
        self.assertEqual(g.mul(f,g.inv(f)),(1,0,0,0,0,0))

    def test_substitution_and_sign_involution(self):
        f=(1,2,3,4,5,6,7)
        self.assertEqual(g.unary(g.unary(f,'N'),'N'),f)
        self.assertEqual(g.unary(f,'S'),(1,0,2,0,3,0,4))
        self.assertEqual(g.unary(g.unary(f,'R'),'R'),f)

    def test_graph_fork_join_order_matters(self):
        f=(1,2,1,3,0,0,0)
        self.assertEqual(g.execute(f,('I','I','I')),g.mul(f,f))
        self.assertNotEqual(g.execute(f,('I','N','S')),g.execute(f,('S','N','I')))
        self.assertEqual(len(g.GRAPHS),64)

    def test_training_never_generates_heldout(self):
        d=g.data(include_test=False,train_count=12,dev_count=8)
        self.assertEqual(set(d),{'train','dev'})
        self.assertTrue(all(c.graph in g.TRAIN_GRAPHS for c in d['train']+d['dev']))
        self.assertTrue(all(c.graph[2]=='I' and 'S' not in c.graph for c in d['train']))
        self.assertTrue(all(c.visible not in {x.visible for x in d['train']} for c in d['dev']))

    def test_heldout_distinct_syntax_families(self):
        d=g.data(train_count=12,dev_count=8)
        self.assertEqual(len(d['heldout_graph']),80)
        self.assertTrue(all(c.graph not in g.TRAIN_GRAPHS for c in d['heldout_graph']))
        self.assertEqual(len(d['out_of_grammar']),20)
        self.assertEqual(len(d['late_collision']),16)

    def test_exact_search_accepts_all_heldout_graphs(self):
        d=g.data(train_count=12,dev_count=8)
        self.assertTrue(all(g.search(c)['graph'] is not None for c in d['heldout_graph']))
        self.assertTrue(all(g.search(c)['graph'] is None for c in d['out_of_grammar']))

    def test_prefix_collision_verifier_boundary(self):
        d=g.data(train_count=12,dev_count=8)['late_collision']
        for a,mutant in zip(d[::2],d[1::2]):
            self.assertEqual(a.visible,mutant.visible)
            self.assertEqual(a.query,mutant.query)
            self.assertNotEqual(a.future,mutant.future)
            self.assertEqual(g.search(a),g.search(mutant))
            found=g.search(a)
            self.assertEqual(g.score(a,found)['future_correct']+g.score(mutant,found)['future_correct'],1)

    def test_preview_requires_interpreter_access(self):
        c=g.data(include_test=False,train_count=10,dev_count=5)['dev'][0]
        with patch.object(g,'execute',side_effect=RuntimeError('public interpreter')):
            with self.assertRaisesRegex(RuntimeError,'public interpreter'):g.preview_score(c,g.GRAPHS[0])

    def test_preview_budget_and_work_accounting(self):
        c=g.data(include_test=False,train_count=10,dev_count=5)['dev'][0]
        p=g.guided(c,budget=4)
        self.assertLessEqual(p['attempts'],4)
        self.assertEqual(p['preview_calls'],512)
        self.assertGreaterEqual(p['primitive_calls'],512)
        self.assertGreaterEqual(p['work_proxy'],9600)

    def test_parameter_count_and_source_guard(self):
        ck=json.loads((HERE/'results/rrc11_ranker.json').read_text())
        self.assertEqual(len(ck['weights']),4)
        self.assertEqual(len(ck['allowed_supervised_graphs']),9)
        self.assertEqual(ck['train_digest'],g.digest(g.data(include_test=False)['train']))
        self.assertEqual(ck['graph11_sha256'],b.hashes()['graph11.py'])
        self.assertEqual(ck['train11_sha256'],b.hashes()['train11.py'])

    def test_training_reproducibility(self):
        ck=json.loads((HERE/'results/rrc11_ranker.json').read_text())
        self.assertEqual(ck,t.train())

    def test_full_receipt_and_heldout_failure(self):
        rec=json.loads((HERE/'results/rrc11_pilot.json').read_text())
        self.assertEqual(rec['metrics']['heldout_graph']['exact']['correct_future'],80)
        self.assertEqual(rec['metrics']['heldout_graph']['learned_preview']['correct_future'],79)
        self.assertEqual(rec['metrics']['heldout_graph']['preview']['correct_future'],79)
        self.assertEqual(rec['metrics']['late_collision']['exact']['false_accept'],8)
        self.assertEqual(rec['metrics']['out_of_grammar']['exact']['accepted'],0)
        self.assertEqual(rec['metrics']['heldout_graph']['learned_preview'],rec['metrics']['heldout_graph']['preview'])

    def test_invalid_graph_and_input(self):
        with self.assertRaises(ValueError):g.execute((1,1,2),('Q','I','N'))
        with self.assertRaises(ValueError):g.unary((2,1),'R')
        with self.assertRaises(ValueError):g.mul((1,),(1,1))
        with self.assertRaises(ValueError):g.truncate((1,2),4)

if __name__=='__main__':unittest.main()
