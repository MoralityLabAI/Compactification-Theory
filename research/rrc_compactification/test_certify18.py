"""RRC-18 finite universal polynomial certificates and learned search boundary."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

try:
    from . import certify18 as c, learn18 as l, model18 as m, bench18 as b, topology16 as t, graph11 as g
except ImportError:
    import certify18 as c
    import learn18 as l
    import model18 as m
    import bench18 as b
    import topology16 as t
    import graph11 as g

HERE=Path(__file__).resolve().parent


def evaluate_poly(poly, f):
    value=0
    for word,scale in poly.items():
        contribution=scale
        for i,x in enumerate(f[1:]):
            exponent=(word>>(c.BASE_BITS*i))&31
            contribution*=x**exponent
        value+=contribution
    return value

class PolynomialCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=c.certificate()

    def test_ring_unit_variable_convention(self):
        var=c.variables()
        self.assertEqual(var[0],{0:1})
        for i in range(1,c.T+1):
            self.assertEqual(var[i],{1<<(5*(i-1)):1})
        with self.assertRaises(ValueError):c.variables(13)

    def test_polynomial_arithmetic_and_unit_inverse(self):
        a=[{0:1},{1:1},{}]
        inv=c.poly_inv(a)
        self.assertEqual(inv,[{0:1},{1:-1},{2:1}])
        self.assertEqual(c.convolve(a,inv),[{0:1},{},{}])
        self.assertEqual(c.poly_mul({1:2},{1:3}),{2:6})
        with self.assertRaises(ValueError):c.poly_inv([{},{}])

    def test_independent_concrete_parity_for_all_four_times_64_programs(self):
        rng=random.Random(202611118)
        vars=c.variables()
        for topology in t.TOPOLOGIES:
            for program in g.GRAPHS:
                symbolic=c.execute(vars,program,topology)
                for _ in range(2):
                    coefficients=(1,)+tuple(rng.randint(-3,3) for _ in range(c.T))
                    direct=t.execute(coefficients,program,topology)
                    actual=tuple(evaluate_poly(p,coefficients) for p in symbolic)
                    self.assertEqual(actual,direct,(topology,program))

    def test_all_probe_partitions_equal_exact_polynomial_partitions(self):
        self.assertEqual({p:d['class_count'] for p,d in self.report['topologies'].items()},
                         {'fork':25,'cascade':24,'additive':33,'hadamard':24})
        self.assertTrue(all(d['matches_rrc17_probe_partition'] for d in self.report['topologies'].values()))
        self.assertTrue(all(sum(d['class_sizes'])==64 for d in self.report['topologies'].values()))

    def test_exact_normal_forms_identify_every_class_pair(self):
        x=c.variables()
        for topology in t.TOPOLOGIES:
            groups,_=c.build_classes(topology)
            seen=set()
            signatures=[]
            for group in groups:
                signatures.append(c.polynomial_signature(c.execute(x,group[0],topology)))
                first=signatures[-1]
                for graph in group:
                    self.assertEqual(first,c.polynomial_signature(c.execute(x,graph,topology)))
                    self.assertNotIn(graph,seen)
                    seen.add(graph)
            self.assertEqual(seen,set(g.GRAPHS))
            self.assertEqual(len(set(signatures)),len(groups))

    def test_nonidentity_and_finite_scope(self):
        f=(1,1,1,1,0,0,0,0,0,0,0,0,0)
        self.assertNotEqual(t.execute(f,('I','I','I'),'hadamard'),
                            t.execute(f,('I','I','I'),'fork'))
        self.assertIn('through q^12',self.report['claim_scope'])

class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping=l.class_map()
        cls.data=l.roster(train_per=8,dev_per=3)

    def test_dataset_groups_and_train_topologies(self):
        self.assertEqual({k:len(v) for k,v in self.data.items() if k not in ('train','dev')},l.COUNTS)
        self.assertEqual(len(self.data['train']),32)
        self.assertEqual(len(self.data['dev']),12)
        self.assertEqual({c.topology for c in self.data['train']},set(t.TOPOLOGIES))
        self.assertEqual({c.topology for c in self.data['product_shift']},set(t.TOPOLOGIES))

    def test_training_generation_never_uses_product_generator(self):
        with patch.object(t.b13,'_product_source',side_effect=AssertionError('test family generated')):
            data=l.roster(include_test=False,train_per=6,dev_per=2)
        self.assertEqual(set(data),{'train','dev'})
        self.assertTrue(all(c.hidden_graph in g.GRAPHS for v in data.values() for c in v))

    def test_all_raw_features_are_public_and_shape_stable(self):
        case=self.data['train'][0]
        self.assertEqual(len(l.sequence(case)),13)
        self.assertEqual(len(l.sequence(case)[0]),9)
        self.assertEqual(len(l.flat(case)),56)
        altered=replace(case,hidden_graph=None,future=(1,)*33,identifier='changed')
        self.assertEqual(l.sequence(case),l.sequence(altered))
        self.assertEqual(l.flat(case),l.flat(altered))

    def test_finite_class_labels_and_unique_representatives(self):
        labels=l.labels(self.data['train'],self.mapping)
        self.assertEqual(len(labels),32)
        for c,label in zip(self.data['train'],labels):
            self.assertEqual(self.mapping[c.topology]['mapping'][c.hidden_graph],g.GRAPHS[label])
            reps=self.mapping[c.topology]['representatives']
            self.assertEqual(len(reps),len(set(reps)))
            self.assertIn(g.GRAPHS[label],reps)

    def test_no_visible_train_test_alias_and_explicit_collision(self):
        training={c.visible for c in self.data['train']}
        for split in ('signed_holdout','product_shift','out_of_grammar'):
            self.assertFalse(training&{c.visible for c in self.data[split]})
        for a,bad in zip(self.data['future_collision'][::2],self.data['future_collision'][1::2]):
            self.assertEqual(a.visible,bad.visible)
            self.assertEqual(a.query,bad.query)
            self.assertNotEqual(a.future,bad.future)
            self.assertEqual(len(t.matching_graphs(a)),len(t.matching_graphs(bad)))

    def test_negative_examples_rejected_by_full_exact_grammar(self):
        for case in self.data['out_of_grammar']:
            self.assertEqual(t.matching_graphs(case),())

class ModelAndWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.hashes=b.checkpoints()
        cls.case=l.roster(include_test=False,train_per=3,dev_per=2)['dev'][0]
        cls.classes=l.class_map()

    def test_parameter_count_and_training_hashes(self):
        self.assertEqual(m.parameter_count('recurrent'),6240)
        self.assertEqual(m.parameter_count('feedforward'),6235)
        for kind,ck in self.models.items():
            self.assertEqual(ck['source_sha256'],b.t18.sources())
            self.assertEqual(ck['parameter_count'],m.parameter_count(kind))
            self.assertEqual(ck['inference_candidate_executions'],0)

    def test_direct_neural_forward_uses_no_graph_interpreter(self):
        original=t.execute
        with patch.object(t,'execute',side_effect=AssertionError('illegal oracle access')):
            for kind,checkpoint in self.models.items():
                self.assertEqual(len(m.logits(self.case,checkpoint)),64)
        with patch.object(t,'execute',wraps=original) as executed:
            scores=m.logits(self.case,self.models['recurrent'])
        self.assertEqual(executed.call_count,0)
        proposed=l.candidates(scores,self.case.topology,self.classes)
        self.assertEqual(len(proposed),12)
        self.assertEqual(len(set(proposed)),12)

    def test_every_paid_query_evaluates_both_demos(self):
        case=self.case
        order=l.mass_order(case.topology,self.classes)
        original=t.execute
        with patch.object(t,'execute',wraps=original) as executed:
            response=l.search(case,order,budget=4)
        self.assertEqual(executed.call_count,2*len(response['trace']))
        self.assertEqual(response['integer_loop_proxy'],sum(2*t.graph_cost(p,case.topology) for p in response['trace']))
        with self.assertRaises(ValueError):l.search(case,order,budget=0)

    def test_correct_canonical_mask_top12_and_mass_prior(self):
        for topology in t.TOPOLOGIES:
            order=l.mass_order(topology,self.classes)
            self.assertEqual(set(order),set(self.classes[topology]['representatives']))
            self.assertEqual(len(order),len(c.build_classes(topology)[0]))

    def test_logits_parity_if_torch_available(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch optional for inference replay')
        torch.set_num_threads(1)
        for kind in ('recurrent','feedforward'):
            ck=self.models[kind]
            net=b.t18.modules()[kind]()
            net.load_state_dict({k:torch.tensor(v) for k,v in ck['state_dict'].items()})
            if kind=='recurrent':
                x=torch.tensor([l.sequence(self.case)],dtype=torch.float32)
            else:
                x=torch.tensor([l.flat(self.case)],dtype=torch.float32)
            with torch.no_grad():true=net(x)[0].tolist()
            portable=m.logits(self.case,ck)
            self.assertLess(max(abs(x-y) for x,y in zip(true,portable)),3e-5)

    def test_first_candidate_future_collision_identical_proposals(self):
        c=l.roster(include_test=False,train_per=3,dev_per=2)['dev'][1]
        twin=replace(c,identifier='fake',future=(1,)*33,hidden_graph=None)
        for kind,ck in self.models.items():
            a=l.candidates(m.logits(c,ck),c.topology,self.classes)
            v=l.candidates(m.logits(twin,ck),twin.topology,self.classes)
            self.assertEqual(a,v)
            self.assertEqual(l.search(c,a),l.search(twin,v))

    def test_source_bound_receipt_contract(self):
        obj=json.loads((HERE/'results'/'rrc18_pilot.json').read_text())
        self.assertEqual(obj['source_sha256'],b.source_hashes())
        self.assertEqual(obj['checkpoint_sha256'],self.hashes)
        self.assertEqual(obj['parameters']['query_budget'],12)
        self.assertEqual(obj['parameters']['neural_candidate_previews'],0)
        self.assertEqual(obj['certificate']['topologies']['fork']['class_count'],25)
        self.assertEqual(obj['scores']['signed_holdout']['complete_certified_quotient']['accepted'],240)
        self.assertEqual(obj['scores']['out_of_grammar']['complete_64']['accepted'],0)
        self.assertEqual(obj['scores']['future_collision']['complete_64']['false_accept'],8)

if __name__=='__main__':unittest.main()
