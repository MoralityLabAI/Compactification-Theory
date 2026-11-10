"""RRC-22: exact arithmetic, decision parity, isolation, and physical-receipt tests."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

try:
    from . import meter22 as met, bench22 as b, bench21 as old, select21 as d
    from . import learn18 as l, topology16 as t, graph11 as g
except ImportError:
    import meter22 as met
    import bench22 as b
    import bench21 as old
    import select21 as d
    import learn18 as l
    import topology16 as t
    import graph11 as g

HERE=Path(__file__).resolve().parent


class ArithmeticInstrumentation(unittest.TestCase):
    def test_all_public_graphs_and_wirings_exact_parity(self):
        rng=random.Random(2026111022)
        for degree in (4,12):
            for _ in range(2):
                f=(1,)+tuple(rng.randint(-4,4) for _ in range(degree))
                for top in t.TOPOLOGIES:
                    for graph in g.GRAPHS:
                        meter=met.Meter()
                        self.assertEqual(met.execute(f,graph,top,meter),t.execute(f,graph,top))
                        self.assertGreaterEqual(meter.max_intermediate_bits,max(x.bit_length() for x in f))

    def test_large_integer_width_measurement_and_exact_execution(self):
        f=(1,2**74+7,-2**81,1,0,1,1)
        m=met.Meter()
        x=met.execute(f,('R','N','R'),'fork',m)
        self.assertEqual(x,t.execute(f,('R','N','R'),'fork'))
        self.assertGreaterEqual(m.max_operand_bits,74)
        self.assertGreaterEqual(m.max_intermediate_bits,81)
        self.assertGreater(m.mul_operand_bit_product,0)
        self.assertGreater(m.add_operand_max_bits,0)

    def test_meter_keeps_operation_kinds_distinct(self):
        f=(1,3,2,1,0,0)
        for topo in t.TOPOLOGIES:
            m=met.Meter();met.execute(f,('I','I','I'),topo,m)
            self.assertGreater(m.max_intermediate_bits,0)
            self.assertGreater(m.integer_comparisons,-1)
        m=met.Meter();met.execute(f,('I','I','I'),'additive',m)
        self.assertEqual(m.multiplications,0)
        self.assertEqual(m.additions,len(f)-1)
        m=met.Meter();met.execute(f,('I','I','I'),'hadamard',m)
        self.assertEqual(m.multiplications,len(f)-1)

    def test_verifier_trace_parity_even_on_rejection(self):
        data=d.dataset(include_test=False,reference_per=2,calibration_per=2)
        c=data['calibration'][0]
        for order in ((('I','I','I'),),g.GRAPHS[:12]):
            reference=l.search(c,order,budget=len(order))
            candidate,meter=met.metered_trace(c,reference['trace'])
            self.assertEqual(reference['graph'],candidate)
            self.assertGreater(meter.integer_comparisons,0)

    def test_bad_input_and_graph_rejected(self):
        with self.assertRaises(ValueError):met.execute((2,1),('I','I','I'),'fork')
        with self.assertRaises(ValueError):met.execute((1,2),('Q','I','I'),'fork')
        with self.assertRaises(ValueError):met.execute((1,2.,3),('I','I','I'),'fork')


class RouteAndCostAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.cal,cls.classes,cls.all_cases,_,cls.data=b.setup('selective_full')
        cls.cases=[cls.data['signed'][0],cls.data['product'][0],
                   cls.data['rational'][0],cls.data['cubic_sparse'][0]]

    def test_old_deployed_policy_same_candidate_order_and_budget(self):
        for c in self.cases:
            logits=old.neural_logits(c,self.models)
            response=old.router(c,self.cal,self.models,self.classes,logits=logits)
            for arm in ('neural12','selective12','selective_full','ood_only_full','full_certified'):
                proposal=b.decide(c,arm,self.models,self.cal,self.classes)
                if arm=='full_certified':
                    expected=old.choose(c,arm,self.classes,None,None,self.cal)
                else:
                    expected=old.choose(c,arm,self.classes,logits,response,self.cal)
                self.assertEqual((proposal[0],proposal[1],proposal[2]),expected)

    def test_complete_solver_requires_zero_neural_inference_and_no_calibration(self):
        c=self.cases[0]
        with patch.object(b.b,'neural_logits',side_effect=AssertionError('model requested')):
            with patch.object(b.b,'checkpoints',side_effect=AssertionError('weights requested')):
                order,budget,reason,model_used,deferred=b.decide(c,'full_certified',None,None,self.classes)
                self.assertFalse(model_used)
                self.assertFalse(deferred)
                self.assertEqual(budget,len(self.classes[c.topology]['representatives']))

    def test_ood_fast_path_does_not_infer_neural_logits(self):
        c=next(c for fam in d.NOVEL for c in self.data[fam]
               if b.r.calibrated_distance(self.cal,c)[1]<=self.cal['ood_alpha'])
        with patch.object(b.b,'neural_logits',side_effect=AssertionError('model ran')):
            decision=b.decide(c,'selective_full',self.models,self.cal,self.classes)
        self.assertEqual(decision[2],'ood')
        self.assertFalse(decision[3])
        self.assertTrue(decision[4])

    def test_fixed_and_expanded_query_limits(self):
        for c in self.cases:
            for arm in b.ARMS:
                z=b.evaluate(c,arm,self.models if arm!='full_certified' else None,
                             self.cal if arm!='full_certified' else None,self.classes)
                self.assertEqual(len(set(z['trace'])),len(z['trace']))
                self.assertLessEqual(len(z['trace']),33 if arm in ('full_certified','selective_full','ood_only_full') else 12)
                self.assertEqual(z['integer_loop_proxy'],sum(2*t.graph_cost(p,c.topology) for p in z['trace']))
                if z['graph'] is not None:
                    self.assertEqual(t.certificate(c,z['graph'])['accepted'],1)

    def test_future_and_hidden_label_blindness(self):
        c=self.cases[0]
        alt=replace(c,hidden_graph=None,future=(1,)*33,identifier='tampered')
        for arm in b.ARMS:
            a=b.evaluate(c,arm,self.models if arm!='full_certified' else None,
                         self.cal if arm!='full_certified' else None,self.classes)
            z=b.evaluate(alt,arm,self.models if arm!='full_certified' else None,
                         self.cal if arm!='full_certified' else None,self.classes)
            self.assertEqual((a['graph'],a['trace'],a['route']),(z['graph'],z['trace'],z['route']))

    def test_parent_receipt_exact_results_and_integer_bits(self):
        obj=json.loads((HERE/'results'/'rrc22_deterministic.json').read_text())
        self.assertEqual(obj['source_sha256'],b.source_hashes())
        self.assertEqual(obj['parent_receipt_sha256'],b.sha256((HERE/'results'/'rrc21_pilot.json').read_bytes()).hexdigest())
        for arm,expected in (('full_certified',480),('selective_full',457),('neural12',374)):
            rows=[obj['per_arm_family'][arm][k] for k in d.KNOWN+d.NOVEL]
            self.assertEqual(sum(x['future_correct'] for x in rows),expected)
            self.assertGreater(sum(x['ops']['mul_operand_bit_product'] for x in rows),0)
            self.assertGreater(max(x['max_intermediate_bits'] for x in rows),10)
        self.assertEqual(sum(obj['per_arm_family']['selective_full'][k]['deferred'] for k in d.KNOWN+d.NOVEL),311)

    def test_real_measurements_are_unasserted_and_machine_labeled(self):
        obj=json.loads((HERE/'results'/'rrc22_reference_profile.json').read_text())
        self.assertEqual(obj['schema'],'rrc22.physical_profile.v1')
        self.assertEqual(set(obj['timings']),set(b.ARMS))
        self.assertEqual(set(obj['memory']),set(b.ARMS))
        self.assertEqual(obj['source_sha256'],b.source_hashes())
        self.assertEqual(obj['protocol']['rounds'],3)
        for arm,metrics in obj['timings'].items():
            self.assertEqual(metrics['n_measurements'],508*3)
            self.assertGreater(metrics['end_to_end_wall']['total_ms'],0)
            self.assertGreater(obj['memory'][arm]['peak_rss_kib'],0)
            self.assertEqual(obj['memory'][arm]['case_count'],508)
            self.assertLessEqual(obj['memory'][arm]['tracemalloc_sample_cases'],64)

    def test_worker_reports_real_peak_and_memory_sample(self):
        worker=b.memory_worker('full_certified',cases_limit=2)
        self.assertEqual(worker['case_count'],2)
        self.assertEqual(worker['tracemalloc_sample_cases'],2)
        self.assertGreater(worker['peak_rss_kib'],0)
        self.assertGreater(worker['peak_traced_python_bytes'],0)


if __name__=='__main__':unittest.main()
