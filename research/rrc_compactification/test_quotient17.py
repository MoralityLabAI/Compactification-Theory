"""RRC-17 exact arithmetic, randomized quotient, budget and replay gates."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

try:
    from . import quotient17 as q, bench17, topology16 as b, graph11 as g
except ImportError:
    import quotient17 as q
    import bench17
    import topology16 as b
    import graph11 as g

HERE = Path(__file__).resolve().parent


class QuotientAlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quotients,cls.orders = bench17.orderings()

    def test_partition_is_exhaustive_and_nonoverlapping(self):
        sizes={'fork':25,'cascade':24,'additive':33,'hadamard':24}
        for topology,quotient in self.quotients.items():
            flattened=[p for group in quotient.classes for p in group]
            self.assertEqual(set(flattened),set(g.GRAPHS))
            self.assertEqual(len(flattened),64)
            self.assertEqual(len(quotient.classes),sizes[topology])
            self.assertEqual(quotient.total_probe_graph_executions,256)
            self.assertEqual(len(quotient.probes_hash),64)

    def test_representatives_match_fresh_exact_integer_sources(self):
        rng=random.Random(q.SEED+710)
        for topology,quotient in self.quotients.items():
            for x in (g.source(rng)[:13] for _ in range(3)):
                for group in quotient.classes:
                    signatures={b.execute(x,p,topology) for p in group}
                    self.assertEqual(len(signatures),1,(topology,group))

    def test_stress_probes_from_new_seed_have_zero_detected_collisions(self):
        for x in self.quotients.values():
            self.assertGreater(q.verify_probe_stability(x),100)

    def test_offline_builder_does_not_construct_or_read_test_roster(self):
        with patch.object(q,'roster',side_effect=AssertionError('test roster used')):
            with patch.object(b,'matching_graphs',side_effect=AssertionError('target used')):
                self.assertEqual(len(q.build_quotient('hadamard').classes),24)

    def test_fixed_probe_source_is_independent_of_graph_labels(self):
        self.assertEqual(q.random_probes(),q.random_probes())
        self.assertNotEqual(q.random_probes(seed=q.SEED),q.random_probes(seed=q.SEED+1))
        self.assertEqual(len(q.random_probes()),q.PROBES)
        self.assertEqual(len(q.random_probes()[0]),q.PREFIX+1)
        self.assertTrue(all(x[0]==1 and all(0<=v<q.PRIME for v in x[1:])
                            for x in q.random_probes()))

    def test_static_mass_order_is_optimal_for_uniform_syntax_boolean_feedback(self):
        for quotient in self.quotients.values():
            actual=sum(len(group) for group in quotient.sorted_classes(mode='mass')[:q.BUDGET])
            optimal=sum(sorted((len(group) for group in quotient.classes),reverse=True)[:q.BUDGET])
            self.assertEqual(actual,optimal)
            self.assertEqual(len(bench17.orderings()[1][quotient.topology]['quotient_mass12']),
                             len(quotient.classes))

    def test_prime_probe_false_merge_union_bound(self):
        # Schwartz-Zippel, conditional on two graphs differing over F_p.
        bound=len(q.TOPOLOGIES)*2016*((2*q.PREFIX)/q.PRIME)**q.PROBES
        self.assertLess(bound,2e-27)

    def test_invalid_search_and_quotient_arguments(self):
        with self.assertRaises(ValueError):q.build_quotient('unseen')
        with self.assertRaises(ValueError):q.random_probes(count=0)
        with self.assertRaises(ValueError):self.quotients['hadamard'].sorted_classes(mode='wrong')
        c=q.roster()['hadamard_uniform'][0]
        with self.assertRaises(ValueError):q.search(c,g.GRAPHS,budget=0)
        with self.assertRaises(ValueError):q.search(c,g.GRAPHS,budget=65)


class RosterAndBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=q.roster()
        cls.quotients,cls.orders=bench17.orderings()
        cls.models,cls.old,cls.initializer,cls.checkpoint_sha=bench17.bench16.checkpoints()

    def test_case_counts_and_no_unadvertised_duplicate_signatures(self):
        self.assertEqual({name:len(cases) for name,cases in self.data.items()},q.COUNTS)
        prior=set()
        for split,cases in self.data.items():
            for case in cases:
                self.assertEqual(case.topology,q.SPLITS[split])
                if case.visible in prior:
                    self.assertEqual(split,'hadamard_collision')
                prior.add(case.visible)

    def test_hadamard_novel_source_is_old_dsl_unexpressible(self):
        for case in self.data['hadamard_novel']:
            self.assertTrue(b.matching_graphs(case))
            self.assertTrue(all(not b.matching_graphs(case,old)
                                for old in b.TRAIN_TOPOLOGIES))

    def test_negative_roster_is_finite_class_inconsistent(self):
        self.assertTrue(all(not b.matching_graphs(case)
                            for case in self.data['hadamard_negative']))
        self.assertTrue(all(q.search(case,g.GRAPHS,budget=64)['graph'] is None
                            for case in self.data['hadamard_negative']))

    def test_randomized_and_neural_arms_do_not_look_at_hidden_future(self):
        item=self.data['hadamard_uniform'][0]
        mutant=replace(item,future=(1,)*len(item.future),hidden_graph=None,identifier='secret')
        for arm in bench17.ARMS:
            p=bench17.prediction(item,arm,self.orders,self.models,self.old,self.initializer)
            m=bench17.prediction(mutant,arm,self.orders,self.models,self.old,self.initializer)
            self.assertEqual((p['graph'],p['trace']),(m['graph'],m['trace']))

    def test_all_paid_queries_correspond_to_two_interpreter_executions(self):
        item=self.data['hadamard_novel'][0]
        original=b.execute
        for arm in ('quotient_mass12','first12','quotient_all','frozen_gru'):
            with patch.object(b,'execute',wraps=original) as called:
                result=bench17.prediction(item,arm,self.orders,self.models,self.old,self.initializer)
            self.assertEqual(called.call_count,2*result['queries'])
            self.assertEqual(result['primitive_calls'],8*result['queries'])
            self.assertEqual(result['integer_loop_proxy'],sum(2*b.graph_cost(x,item.topology)
                                                              for x in result['trace']))
            self.assertEqual(len(set(result['trace'])),len(result['trace']))
            self.assertLessEqual(result['queries'],64 if arm=='quotient_all' else 12)

    def test_no_free_verified_final_guess_after_budget(self):
        item=self.data['hadamard_uniform'][0]
        guess=q.search(item,g.GRAPHS,budget=1)
        self.assertEqual(guess['queries'],1)
        self.assertEqual(guess['graph'] is not None, g.GRAPHS[0] in b.matching_graphs(item))
        if guess['graph'] is not None:
            self.assertTrue(b.feedback(item,guess['graph'])['accepted'])

    def test_collision_pairs_have_identical_visible_prediction_paths(self):
        for a,bad in zip(self.data['hadamard_collision'][::2],
                         self.data['hadamard_collision'][1::2]):
            self.assertEqual(a.visible,bad.visible)
            self.assertEqual(a.query,bad.query)
            self.assertNotEqual(a.future,bad.future)
            for arm in ('quotient_mass12','quotient_random12','frozen_gru','exact_64'):
                one=bench17.prediction(a,arm,self.orders,self.models,self.old,self.initializer)
                two=bench17.prediction(bad,arm,self.orders,self.models,self.old,self.initializer)
                self.assertEqual((one['trace'],one['graph']),(two['trace'],two['graph']))

    def test_reused_frozen_models_are_source_hash_locked(self):
        self.assertEqual(self.models['recurrent']['parameter_count'],7497)
        self.assertEqual(self.models['feedforward']['parameter_count'],7549)
        self.assertEqual(self.models['recurrent']['source_sha256'],bench17.bench16.train16.source_hashes())
        self.assertEqual(len(self.checkpoint_sha),6)

    def test_registered_receipt_and_exact_upper_controls(self):
        saved=json.loads((HERE/'results'/'rrc17_pilot.json').read_text())
        self.assertEqual(saved['source_sha256'],bench17.source_hashes())
        self.assertEqual(saved['rrc16_checkpoint_sha256'],self.checkpoint_sha)
        self.assertEqual(saved['roster_sha256'],{k:q.hash_roster(v) for k,v in self.data.items()})
        self.assertEqual(saved['probe_build_total_executions'],1024)
        self.assertEqual(saved['budget'],12)
        for group in ('hadamard_uniform','hadamard_novel','hadamard_product',
                      'fork_uniform','cascade_uniform','additive_uniform'):
            self.assertEqual(saved['scores'][group]['quotient_all']['accepted'],q.COUNTS[group])
            self.assertEqual(saved['scores'][group]['exact_64']['accepted'],q.COUNTS[group])
        self.assertEqual(saved['scores']['hadamard_negative']['exact_64']['accepted'],0)
        self.assertEqual(saved['scores']['hadamard_collision']['exact_64']['false_accept'],8)
        self.assertTrue(all(x['identical_traces'] for x in saved['late_collisions'].values()))
        self.assertEqual(len(saved['independent_seed_replications']),3)


if __name__=='__main__':unittest.main()
