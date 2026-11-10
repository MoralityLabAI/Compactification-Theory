"""RRC-20 exact-series, calibration blindness, cost and source-lock tests."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

try:
    from . import family20 as f,gate20 as gate,train20 as tr,bench20 as bench
    from . import graph11 as g,topology16 as t,learn18 as l,model18 as m
except ImportError:
    import family20 as f
    import gate20 as gate
    import train20 as tr
    import bench20 as bench
    import graph11 as g
    import topology16 as t
    import learn18 as l
    import model18 as m
HERE=Path(__file__).resolve().parent


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups=f.roster(known_per=3,novel_per=4,negatives=3,pairs=2)

    def test_generator_unit_integrality_and_horizon(self):
        rng=random.Random(202611120)
        for family in f.FAMILIES:
            for _ in range(6):
                v=f.source(rng,family)
                self.assertEqual((len(v),v[0]),(33,1))
                self.assertTrue(all(type(x) is int for x in v))
        with self.assertRaises(ValueError):f.source(rng,'not_a_family')

    def test_compositional_sources_exact(self):
        rng=random.Random(2001);a=f.old.source(rng,'product')
        rng=random.Random(2001);twist=f.source(rng,'twisted_product')
        self.assertEqual(twist,tuple(v*(-1 if n%2 else 1) for n,v in enumerate(a)))
        for name in ('rational_product','mixed_signed_product','cubic_sparse'):
            v=f.source(random.Random(777),name)
            self.assertEqual(v[0],1)
            self.assertEqual(g.mul(v,g.inv(v)),(1,)+(0,)*32)

    def test_new_sources_absent_from_calibration(self):
        with patch.object(f,'source',side_effect=AssertionError('test source called during calibration')):
            dev=f.calibration()
        self.assertEqual(len(dev),420)
        self.assertTrue(all(z.identifier.split('-')[2] in f.KNOWN for z in dev))

    def test_roster_counts_and_topologies(self):
        d=self.groups
        self.assertEqual({k:len(v) for k,v in d.items()},
                         {**{z:(12 if z in f.KNOWN else 16) for z in f.FAMILIES},
                          'negative':3,'collision':4})
        for family in f.FAMILIES:
            self.assertEqual({top:sum(c.topology==top for c in d[family]) for top in t.TOPOLOGIES},
                             {top:(3 if family in f.KNOWN else 4) for top in t.TOPOLOGIES})

    def test_no_visible_aliases_with_prior_exposed_roster(self):
        seen={case.visible for cases in f.old.roster().values() for case in cases}
        for split,cases in self.groups.items():
            if split=='collision':
                for a,b in zip(cases[::2],cases[1::2]):
                    self.assertEqual((a.visible,a.query),(b.visible,b.query))
                    self.assertNotEqual(a.future,b.future)
                    self.assertNotIn(a.visible,seen)
            else:
                self.assertFalse(any(c.visible in seen for c in cases))

    def test_negative_and_full_class_consistency(self):
        for c in self.groups['negative']:
            self.assertFalse(t.matching_graphs(c))
        for family in f.FAMILIES:
            for c in self.groups[family][:4]:
                self.assertTrue(t.matching_graphs(c))
                self.assertEqual(t.certificate(c,c.hidden_graph)['accepted'],1)

    def test_exact_future_collision_bound(self):
        for a,b in zip(self.groups['collision'][::2],self.groups['collision'][1::2]):
            gr=t.matching_graphs(a)[0]
            self.assertEqual(sum(t.certificate(x,gr)['future_correct'] for x in (a,b)),1)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,_,cls.gates,cls.digests=bench.checkpoints()
        cls.classes=l.class_map()
        cls.sample=f.roster(known_per=2,novel_per=2,negatives=2,pairs=1)['rational_product'][0]

    def test_parameter_and_checkpoint_locks(self):
        self.assertEqual(self.gates['adaptive']['parameter_count'],60)
        self.assertEqual(self.gates['constant']['parameter_count'],4)
        self.assertEqual(self.gates['shuffled']['parameter_count'],60)
        self.assertEqual(self.gates['adaptive']['calibration_digest'],f.old.digest(f.calibration()))
        self.assertEqual(set(self.gates),set(tr.ROLES))
        self.assertEqual(self.gates['adaptive']['source_sha256'],tr.source_hashes())

    def test_feature_shape_and_hidden_label_blindness(self):
        c=self.sample
        changed=replace(c,hidden_graph=None,future=(1,)*len(c.future),identifier='changed',family='unlabeled')
        self.assertEqual(gate.source_features(c),gate.source_features(changed))
        self.assertEqual(len(gate.source_features(c)),14)
        for ck in self.gates.values():
            self.assertEqual(gate.weights(c,ck),gate.weights(changed,ck))

    def test_gate_weights_and_distributions(self):
        c=self.sample
        for ck in self.gates.values():
            weights=gate.weights(c,ck)
            self.assertEqual(len(weights),4)
            self.assertAlmostEqual(sum(weights),1,places=12)
            self.assertTrue(all(z>0 for z in weights))
        logits={role:m.logits(c,self.models[role]) for role in gate.EXPERTS[:3]}
        probs=gate.expert_distributions(c.topology,logits,self.classes)
        self.assertEqual(len(probs),4)
        self.assertEqual(len(probs[0]),len(self.classes[c.topology]['representatives']))
        for row in probs:self.assertAlmostEqual(sum(row),1,places=12)

    def test_inference_does_not_call_graph_interpreter(self):
        c=self.sample
        with patch.object(t,'execute',side_effect=AssertionError('candidate execution leaked to neural inference')):
            logits={role:m.logits(c,self.models[role]) for role in gate.EXPERTS[:3]}
            for kind in tr.ROLES:
                order=gate.order(c,logits,self.classes,self.gates[kind])
                self.assertEqual(len(order),len(self.classes[c.topology]['representatives']))

    def test_paid_verifier_executes_two_demos_and_costs_work(self):
        c=self.sample
        logits={role:m.logits(c,self.models[role]) for role in gate.EXPERTS[:3]}
        proposed=bench.candidate_order(c,'adaptive_mix12',logits,self.classes,self.gates)
        with patch.object(t,'execute',wraps=t.execute) as called:
            result=l.search(c,proposed,budget=12)
        self.assertEqual(called.call_count,2*len(result['trace']))
        self.assertEqual(result['integer_loop_proxy'],
                         sum(2*t.graph_cost(z,c.topology) for z in result['trace']))
        self.assertLessEqual(len(result['trace']),12)

    def test_mix_budget_and_model_compute(self):
        c=self.sample
        logits={role:m.logits(c,self.models[role]) for role in gate.EXPERTS[:3]}
        for arm in bench.ARMS:
            order=bench.candidate_order(c,arm,logits,self.classes,self.gates)
            self.assertEqual(len(order),len(set(order)))
            self.assertLessEqual(len(order),33)
        self.assertEqual(bench.macs('adaptive_mix12'),112608)
        self.assertEqual(bench.macs('constant_mix12'),112552)
        self.assertEqual(bench.macs('balanced_gru12'),53216)
        self.assertEqual(bench.macs('mass12'),0)

    def test_shuffled_gate_only_trains_final_gate(self):
        for role in tr.ROLES:
            obj=self.gates[role]
            self.assertTrue(obj['base_models_frozen'])
            self.assertEqual(obj['checkpoint_sha256'],{k:v for k,v in self.digests.items() if not k.startswith('rrc20_')})
        self.assertEqual(self.gates['shuffled']['calibration_examples'],420)

    def test_invalid_gate_input_rejected(self):
        c=self.sample
        logits={role:m.logits(c,self.models[role]) for role in gate.EXPERTS[:3]}
        with self.assertRaises(ValueError):gate.order(c,logits,self.classes,fixed=(-1,1,1,1))
        with self.assertRaises(ValueError):gate.order(c,logits,self.classes)
        with self.assertRaises(ValueError):bench.candidate_order(c,'nonsense',logits,self.classes,self.gates)

    def test_portable_gate_parity_if_torch_installed(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch optional')
        torch.set_num_threads(1)
        for kind in ('adaptive','shuffled','constant'):
            ck=self.gates[kind]
            x=torch.tensor(gate.source_features(self.sample),dtype=torch.float64)
            bias=torch.tensor(ck['bias'],dtype=torch.float64)
            pred=torch.softmax(bias if kind=='constant' else
                               bias+torch.tensor(ck['matrix'],dtype=torch.float64)@x,dim=0)
            self.assertLess(max(abs(a-b) for a,b in zip(pred.tolist(),gate.weights(self.sample,ck))),1e-13)

    def test_training_never_constructs_test_roster(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch optional')
        with patch.object(f,'roster',side_effect=AssertionError('test roster reached trainer')):
            records=tr.train(steps=2)
        self.assertEqual(set(records),set(tr.ROLES))
        self.assertTrue(all(c['test_families_used'] is False for c in records.values()))


class ReceiptTests(unittest.TestCase):
    def test_receipt_source_hashes_and_primary_scores(self):
        result=json.loads((HERE/'results'/'rrc20_pilot.json').read_text())
        self.assertEqual(result['source_sha256'],bench.source_hashes())
        self.assertEqual(result['parameters']['budget'],12)
        self.assertEqual(result['summaries']['novel']['adaptive_mix12']['correct_future'],301)
        self.assertEqual(result['summaries']['novel']['uniform_mix12']['correct_future'],318)
        self.assertEqual(result['summaries']['known']['adaptive_mix12']['correct_future'],131)
        self.assertEqual(result['scores']['negative']['complete_classes']['accepted'],0)
        self.assertEqual(result['scores']['collision']['complete_classes']['false_accept'],f.COLLISION_PAIRS)
        self.assertTrue(all(v['identical_trace'] for v in result['collision_pairs'].values()))
        self.assertEqual(result['scores']['cubic_sparse']['adaptive_mix12']['false_accept'],1)


if __name__=='__main__':unittest.main()
