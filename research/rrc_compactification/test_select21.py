"""RRC-21 audit: known-only calibration, exact certification, costs, failures."""
from __future__ import annotations
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from . import select21 as d, risk21 as r, train21 as t, bench21 as b
    from . import learn18 as l, topology16 as top, gate20
except ImportError:
    import select21 as d
    import risk21 as r
    import train21 as t
    import bench21 as b
    import learn18 as l
    import topology16 as top
    import gate20

HERE=Path(__file__).resolve().parent


class DataAndConformalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=d.dataset(include_test=False,reference_per=6,calibration_per=4)
        cls.full=d.dataset(reference_per=6,calibration_per=4,known_per=3,novel_per=3,negatives=4,pairs=2)

    def test_strict_calibration_visibility(self):
        self.assertEqual(set(self.reference),{'reference','calibration'})
        self.assertEqual(len(self.reference['reference']),18)
        self.assertEqual(len(self.reference['calibration']),12)
        for name,cases in self.reference.items():
            self.assertEqual({c.family for c in cases},set(d.KNOWN))
            self.assertTrue(all(c.hidden_graph is not None for c in cases))

    def test_calibration_cannot_construct_new_source(self):
        orig=d.source
        def restricted(rng,name):
            if name in d.NOVEL:raise AssertionError('new family visible during calibration')
            return orig(rng,name)
        with patch.object(d,'source',side_effect=restricted):
            a=d.dataset(include_test=False,reference_per=4,calibration_per=3)
        self.assertEqual(set(a),{'reference','calibration'})

    def test_heldout_roster_independent_and_explicit_pairs(self):
        samples=self.full
        self.assertEqual({k:len(v) for k,v in samples.items() if k not in ('reference','calibration')},
                         {**{k:3 for k in d.KNOWN+d.NOVEL},'negative':4,'collision':4})
        ref={c.visible for c in samples['reference']+samples['calibration']}
        self.assertFalse(any(c.visible in ref for k,v in samples.items() if k not in ('reference','calibration') for c in v))
        for a,mutated in zip(samples['collision'][::2],samples['collision'][1::2]):
            self.assertEqual((a.visible,a.query),(mutated.visible,mutated.query))
            self.assertNotEqual(a.future,mutated.future)
            self.assertEqual(sum(top.certificate(c,a.hidden_graph)['future_correct'] for c in (a,mutated)),1)

    def test_source_unit_and_degree(self):
        import random
        rng=random.Random(21421)
        for k in d.KNOWN+d.NOVEL:
            for _ in range(4):
                src=d.source(rng,k)
                self.assertEqual(len(src),d.HORIZON+1)
                self.assertEqual(src[0],1)
                self.assertTrue(all(type(x)==int for x in src))
        with self.assertRaises(ValueError):d.source(rng,'invented')
        with self.assertRaises(ValueError):d.dataset(reference_per=0)

    def test_pvalue_superuniform_rank_formula(self):
        c=[1.,2.,3.,4.]
        self.assertEqual(r.conformal_pvalue(.5,c),1)
        self.assertEqual(r.conformal_pvalue(5.,c),.2)
        self.assertEqual(r.conformal_pvalue(2.,c),.8)
        with self.assertRaises(ValueError):r.conformal_pvalue(1.,[])

    def test_ood_score_future_and_hidden_blind(self):
        x=self.reference['calibration'][0]
        altered=replace(x,hidden_graph=None,future=(1,)*len(x.future))
        norm=r.fit_reference(d.dataset(include_test=False,reference_per=8,calibration_per=2)['reference'])
        self.assertEqual(r.distance(r.features(x),norm),r.distance(r.features(altered),norm))
        self.assertEqual(r.features(x),r.features(altered))

    def test_reference_raises_on_insufficient_cases(self):
        with self.assertRaises(ValueError):r.fit_reference(self.reference['reference'][:3])


class SelectorAndResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.cal,cls.digests=b.checkpoints()
        cls.classes=l.class_map()
        cls.cases=d.dataset(include_test=False,reference_per=4,calibration_per=4)['calibration']

    def test_frozen_source_and_checkpoint_locks(self):
        self.assertEqual(self.cal['source_sha256'],t.source_hashes())
        self.assertEqual(self.cal['base_model_sha256'],{k:self.digests[k] for k in t.MODEL_ROLES})
        self.assertEqual(self.cal['neural_updates'],0)
        self.assertEqual(self.cal['reference_count'],420)
        self.assertEqual(self.cal['calibration_count'],180)
        self.assertEqual(self.cal['calibration_defer_count'],50)
        self.assertEqual(self.cal['calibration_ood_reject_count'],17)

    def test_no_candidate_preview_in_ood_gate_and_neural_router(self):
        c=self.cases[0]
        original=top.execute
        with patch.object(top,'execute',side_effect=AssertionError('candidate graph executed')):
            pred=b.router(c,self.cal,self.models,self.classes)
        self.assertTrue(pred['pvalue']>0)

    def test_ood_gate_skips_all_neural_inference(self):
        novel=d.dataset(reference_per=4,calibration_per=4,known_per=2,novel_per=3,negatives=2,pairs=1)['amplified_signed'][0]
        with patch.object(b,'neural_logits',side_effect=AssertionError('unneeded neural inference')):
            decision=b.router(novel,self.cal,self.models,self.classes)
        self.assertEqual(decision['reason'],'ood')
        self.assertEqual(decision['macs'],0)
        self.assertIsNone(decision['logits'])

    def test_in_domain_router_requires_entropy_without_logits(self):
        with self.assertRaises(ValueError):r.decision(self.cal,self.cases[0],.9,None)

    def test_finite_verified_queries_have_exact_cost_and_two_executions(self):
        c=self.cases[0]
        models=b.neural_logits(c,self.models)
        decision=b.router(c,self.cal,self.models,self.classes,logits=models)
        for arm in b.ARMS:
            order,budget,_=b.choose(c,arm,self.classes,models,decision,self.cal)
            with patch.object(top,'execute',wraps=top.execute) as invoked:
                outcome=l.search(c,order,budget=budget)
            self.assertEqual(invoked.call_count,2*len(outcome['trace']))
            self.assertEqual(outcome['integer_loop_proxy'],
                             sum(2*top.graph_cost(g,c.topology) for g in outcome['trace']))
            self.assertEqual(len(set(outcome['trace'])),len(outcome['trace']))
            if arm not in ('selective_full','ood_only_full','random_full','full_certified'):
                self.assertLessEqual(len(outcome['trace']),d.BUDGET)

    def test_public_future_and_graph_label_do_not_change_decisions(self):
        c=self.cases[0];alt=replace(c,hidden_graph=None,future=(1,)*len(c.future),identifier='fake')
        logits=b.neural_logits(c,self.models)
        a=b.router(c,self.cal,self.models,self.classes,logits=logits)
        z=b.router(alt,self.cal,self.models,self.classes,logits=logits)
        self.assertEqual(a,z)
        for arm in b.ARMS:
            self.assertEqual(b.choose(c,arm,self.classes,logits,a,self.cal),
                             b.choose(alt,arm,self.classes,logits,z,self.cal))
        self.assertEqual(b.visible_hash_uniform(c),b.visible_hash_uniform(alt))

    def test_join_deduplicates(self):
        x=self.classes['fork']['representatives']
        self.assertEqual(b.join_order(x[:6],x[:12]),x[:12])
        self.assertEqual(len(set(b.join_order(x[:6],x[:12]))),12)

    def test_ood_precalibration_does_not_use_heldout_labels(self):
        self.assertFalse(self.cal['calibration_ground_truth_used'])
        self.assertTrue(self.cal['excludes_test_labels'])
        self.assertFalse(self.cal['test_ood_families_constructed'])
        self.assertFalse(self.cal['full_graph_interpreter_previews_during_calibration'])


class ExactReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=json.loads((HERE/'results'/'rrc21_pilot.json').read_text())

    def test_source_bound_receipt(self):
        self.assertEqual(self.saved['source_sha256'],b.source_hashes())
        self.assertEqual(self.saved['parameters']['strict_budget'],12)
        self.assertEqual(self.saved['parameters']['full_symbolic_max_budget'],33)
        self.assertEqual(self.saved['parameters']['inference_model_mac_proxy'],112552)

    def test_holdout_rates_and_known_calibration(self):
        s=self.saved
        self.assertEqual(s['ood_detection']['known_n'],192)
        self.assertEqual(s['ood_detection']['novel_n'],288)
        self.assertGreater(s['ood_detection']['auroc_novel_vs_known'],0.9)
        self.assertEqual(s['calibration']['calibration_defer_count'],50)

    def test_primary_recovery_and_cost_boundaries(self):
        s=self.saved['summaries']['positive']
        self.assertEqual(s['neural12']['future_correct'],374)
        self.assertEqual(s['selective12']['future_correct'],324)
        self.assertEqual(s['selective_full']['future_correct'],457)
        self.assertEqual(s['full_certified']['future_correct'],480)
        self.assertLessEqual(s['selective_full']['future_correct'],s['full_certified']['future_correct'])
        self.assertEqual(s['selective_full']['deferred'],311)

    def test_all_negative_rejections_and_future_collision_limit(self):
        self.assertEqual(self.saved['scores']['negative']['full_certified']['accepted'],0)
        self.assertTrue(all(x['identical_trace'] for x in self.saved['collision_pairs'].values()))
        self.assertEqual(self.saved['collision_pairs']['full_certified']['false_future_acceptance'],6)
        self.assertEqual(self.saved['scores']['sparse']['neural12']['false_future_acceptance'],1)


if __name__=='__main__':unittest.main()
