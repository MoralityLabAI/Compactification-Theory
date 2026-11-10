"""RRC-15 mathematical, no-leakage, query-budget and frozen receipt gates."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

try:
    from . import topology15 as b, bench15, graph11 as g, graph14 as b14, train14
except ImportError:
    import topology15 as b
    import bench15
    import graph11 as g
    import graph14 as b14
    import train14

HERE=Path(__file__).resolve().parent


class TopologyAlgebraTests(TestCase):
    def test_fork_interpreter_parity_with_frozen_graph11(self):
        import random
        rng=random.Random(1212)
        for _ in range(3):
            f=g.source(rng)
            for gr in g.GRAPHS:
                self.assertEqual(b.execute(f,gr,'fork'),g.execute(f,gr))

    def test_additive_unit_and_formal_reciprocal(self):
        f=(1,2,3,4,0,0,0)
        identity=(1,0,0,0,0,0,0)
        self.assertEqual(b.unit_add(f,identity),f)
        self.assertEqual(b.execute(f,('I','I','I'),'additive'),(1,4,6,8,0,0,0))
        result=b.execute(f,('I','N','R'),'additive')
        self.assertEqual(g.mul(result,b.execute(f,('I','N','I'),'additive')),identity)
        with self.assertRaises(ValueError):b.unit_add((2,1),f)

    def test_cascade_is_genuinely_different_for_some_programs(self):
        f=(1,1,2,3,0,0,0,0,0)
        graph=('N','S','I')
        self.assertNotEqual(b.execute(f,graph,'cascade'),b.execute(f,graph,'fork'))
        self.assertEqual(b.execute(f,('I','S','I'),'cascade'),
                         b.execute(f,('I','S','I'),'fork'))

    def test_all_topology_outputs_are_integer_unit_series(self):
        import random
        rng=random.Random(19)
        for topology in b.TOPOLOGIES:
            for graph in g.GRAPHS:
                y=b.execute(g.source(rng),graph,topology)
                self.assertEqual(y[0],1)
                self.assertTrue(all(type(v) is int for v in y))
                self.assertEqual(len(y),b.HORIZON+1)

    def test_operation_cost_and_validation(self):
        self.assertLess(b.graph_cost(('I','I','I'),'additive'),
                        b.graph_cost(('I','I','I'),'fork'))
        self.assertGreater(b.graph_cost(('R','R','R'),'cascade'),
                           b.graph_cost(('I','N','S'),'cascade'))
        with self.assertRaises(ValueError):b.execute((1,1),('I','I','Q'),'cascade')
        with self.assertRaises(ValueError):b.execute((1,1),('I','I','I'),'unknown')
        with self.assertRaises(ValueError):b.graph_cost(('I','I','I'),'fork',-1)


class RosterAndCertificateTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups=b.roster()

    def test_family_counts_and_observable_signature_split(self):
        self.assertEqual({k:len(v) for k,v in self.groups.items()},b.COUNTS)
        seen={}
        for split,cases in self.groups.items():
            for c in cases:
                self.assertEqual(c.topology,b.SPLIT_TOPOLOGY[split])
                if c.visible in seen:
                    self.assertTrue(split.endswith('collision') and seen[c.visible]==split)
                seen[c.visible]=split

    def test_novel_shift_is_unexpressible_in_old_fork_grammar(self):
        for family in ('cascade_novel','cascade_product','additive_novel','additive_product'):
            for c in self.groups[family]:
                self.assertFalse(b.matching_graphs(c,'fork'))
                self.assertTrue(b.matching_graphs(c))
        self.assertTrue(all(b.matching_graphs(c,'fork') for c in self.groups['cascade_alias']))

    def test_new_seed_not_old_training_roster(self):
        known={c.visible for c in b14.dataset(include_test=False)['train']}
        # RRC-14's old Cases did not carry a topology in visible; compare pairs.
        known_pairs={x for x in known}
        self.assertTrue(all(c.visible[1] not in known_pairs
                            for cases in self.groups.values() for c in cases))

    def test_all_negative_cases_are_class_rejected(self):
        for split in ('cascade_negative','additive_negative'):
            for c in self.groups[split]:
                self.assertFalse(b.matching_graphs(c))
                self.assertEqual(b.certify_future(c,None)['accepted'],0)

    def test_future_collision_pairs_share_full_public_view(self):
        for family in ('cascade_collision','additive_collision'):
            for x,y in zip(self.groups[family][::2],self.groups[family][1::2]):
                self.assertEqual(x.visible,y.visible)
                self.assertEqual(x.query,y.query)
                self.assertNotEqual(x.future,y.future)
                self.assertEqual(len(b.matching_graphs(x)),len(b.matching_graphs(y)))
                candidate=b.matching_graphs(x)[0]
                self.assertEqual(sum(b.certify_future(c,candidate)['correct_future'] for c in (x,y)),1)


class PolicyAndReceiptTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.prior,cls.digests=bench15.checkpoint_locks()
        cls.group=b.roster()

    def test_frozen_weights_and_parameter_counts(self):
        self.assertEqual(self.models['recurrent']['parameters'],7353)
        self.assertEqual(self.models['feedforward']['parameters'],7393)
        self.assertEqual(self.prior['parameter_count'],6148)
        self.assertEqual(self.models['recurrent']['source_hashes'],train14.source_hashes())
        self.assertEqual(set(self.group),set(b.COUNTS))

    def test_interpreter_exact_call_count_with_frozen_recurrent(self):
        c=self.group['cascade_novel'][0]
        original=b.execute
        with patch.object(b,'execute',wraps=original) as executed:
            result=bench15.search_edits(c,'frozen_recurrent',self.models,self.prior,budget=3)
            self.assertEqual(executed.call_count,2*result['queries'])
        self.assertLessEqual(result['queries'],3)
        self.assertEqual(result['queries'],len(result['trace']))
        self.assertEqual(len(set(result['trace'])),result['queries'])
        self.assertEqual(result['primitive_calls'],8*result['queries'])
        self.assertEqual(result['integer_loop_proxy'],
                         sum(2*b.graph_cost(gr,c.topology) for gr in result['trace']))

    def test_no_future_or_hidden_label_access_at_inference(self):
        case=self.group['additive_novel'][0]
        alt=replace(case,future=(1,)*len(case.future),hidden_graph=None,identifier='altered')
        for arm in bench15.ARMS:
            x=bench15.predict(case,arm,self.models,self.prior)
            y=bench15.predict(alt,arm,self.models,self.prior)
            self.assertEqual((x['graph'],x['trace'],x['queries']),
                             (y['graph'],y['trace'],y['queries']))

    def test_reversible_edit_and_cached_search_outputs(self):
        c=self.group['cascade_novel'][1]
        x=bench15.predict(c,'frozen_recurrent',self.models,self.prior)
        y=bench15.predict(c,'frozen_recurrent',self.models,self.prior)
        self.assertEqual(x,y)
        self.assertEqual(len(b14.neighborhood(g.GRAPHS[0])),9)
        self.assertTrue(all(b14.reverse_edge(gr,nxt) for gr in g.GRAPHS[:5]
                            for nxt,_,_ in b14.neighborhood(gr)))

    def test_no_test_time_training(self):
        c=self.group['additive_product'][0]
        with patch.object(train14,'fit',side_effect=AssertionError('trainer entered')):
            r=bench15.predict(c,'frozen_recurrent',self.models,self.prior,budget=2)
        self.assertLessEqual(r['queries'],2)

    def test_receipt_source_hash_and_score_contract(self):
        receipt=json.loads((HERE/'results'/'rrc15_pilot.json').read_text())
        self.assertEqual(receipt['source_sha256'],bench15.source_hashes())
        self.assertEqual(receipt['checkpoint_sha256'],self.digests)
        self.assertEqual(receipt['roster_sha256'],{k:b.digest(v) for k,v in self.group.items()})
        self.assertEqual(receipt['parameters']['training_updates'],0)
        self.assertEqual(receipt['parameters']['graph_queries_budget'],12)
        self.assertEqual(receipt['scores']['cascade_novel']['exact_64']['accepted'],48)
        self.assertEqual(receipt['scores']['additive_novel']['exact_64']['accepted'],48)
        self.assertEqual(receipt['scores']['cascade_negative']['exact_64']['accepted'],0)
        self.assertEqual(receipt['scores']['additive_negative']['exact_64']['accepted'],0)
        self.assertEqual(receipt['old_fork_expressible']['cascade_novel'],0)
        self.assertEqual(receipt['old_fork_expressible']['cascade_alias'],24)
        for k,v in receipt['scores'].items():
            self.assertEqual(v['exact_64']['cases'],b.COUNTS[k])
            for arm in bench15.ARMS:
                self.assertLessEqual(v[arm]['accepted'],v[arm]['cases'])
                if arm!='exact_64':self.assertLessEqual(v[arm]['mean_queries'],12)

if __name__=='__main__':
    import unittest
    unittest.main()
