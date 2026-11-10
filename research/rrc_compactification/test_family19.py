"""RRC-19: exact source-family, blind inference and replay tests."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import random
import unittest
from unittest.mock import patch

try:
    from . import family19 as f, train19 as tr, bench19 as b, learn18 as l, model18 as m
    from . import topology16 as t, graph11 as g, certify18
except ImportError:
    import family19 as f
    import train19 as tr
    import bench19 as b
    import learn18 as l
    import model18 as m
    import topology16 as t
    import graph11 as g
    import certify18

HERE=Path(__file__).resolve().parent


class SourceAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=f.roster(train_per=4,dev_per=2,test_per=3)

    def test_all_families_are_exact_unit_series(self):
        rng=random.Random(119)
        for name in f.ALL_FAMILIES:
            for _ in range(10):
                a=f.source(rng,name)
                self.assertEqual(len(a),f.HORIZON+1)
                self.assertEqual(a[0],1)
                self.assertTrue(all(type(v) is int for v in a))
        with self.assertRaises(ValueError):f.source(rng,'invented')

    def test_independent_reciprocal_and_square(self):
        rng=random.Random(42)
        for _ in range(5):
            a=f.source(rng,'rational')
            self.assertEqual(g.mul(a,g.inv(a)),(1,)+(0,)*f.HORIZON)
            s=f.source(rng,'squared')
            self.assertEqual(s[0],1)
            self.assertTrue(all(type(v) is int for v in s))

    def test_train_generator_cannot_touch_ood_source(self):
        original=f.source
        def denied(rng,name):
            if name in f.OOD_FAMILIES:raise AssertionError('OOD source entered training')
            return original(rng,name)
        with patch.object(f,'source',side_effect=denied):
            roster=f.roster(include_test=False,train_per=5,dev_per=2)
        self.assertEqual(set(roster),{'train','dev'})
        self.assertEqual(len(roster['train']),60)
        self.assertEqual(len(roster['dev']),24)
        self.assertEqual({f.source_family(x) for x in roster['train']},set(f.TRAIN_FAMILIES))

    def test_training_is_exactly_cell_balanced(self):
        for split,count in (('train',4),('dev',2)):
            counts={(fam,top):sum(f.source_family(c)==fam and c.topology==top for c in self.data[split])
                    for fam in f.TRAIN_FAMILIES for top in l.TOPOLOGIES}
            self.assertEqual(set(counts.values()),{count})

    def test_holdout_cell_counts_and_no_visible_alias(self):
        self.assertEqual({k:len(v) for k,v in self.data.items() if k not in ('train','dev')},
                         {**{z+'_test':3*len(l.TOPOLOGIES) for z in f.ALL_FAMILIES},
                          'negative':f.NEGATIVE_N,'collision':2*f.COLLISION_PAIRS})
        train={case.visible for case in self.data['train']}
        self.assertFalse(any(case.visible in train for group,cases in self.data.items()
                             if group!='train' for case in cases))
        self.assertEqual(len(self.data['train']),4*len(f.TRAIN_FAMILIES)*len(l.TOPOLOGIES))

    def test_unseen_source_no_hidden_program_for_neural_features(self):
        for split in ('rational_test','squared_test'):
            c=self.data[split][0]
            other=replace(c,hidden_graph=None,future=(1,)*len(c.future),identifier='other')
            self.assertEqual(l.sequence(c),l.sequence(other))
            self.assertEqual(l.flat(c),l.flat(other))
            self.assertEqual(len(l.sequence(c)),13)
            self.assertEqual(len(l.flat(c)),56)

    def test_bounded_negative_and_collision(self):
        self.assertTrue(all(t.matching_graphs(case)==() for case in self.data['negative']))
        for x,y in zip(self.data['collision'][::2],self.data['collision'][1::2]):
            self.assertEqual((x.visible,x.query),(y.visible,y.query))
            self.assertNotEqual(x.future,y.future)
            gr=t.matching_graphs(x)[0]
            self.assertEqual(sum(t.certificate(c,gr)['future_correct'] for c in (x,y)),1)

    def test_positive_source_programs_are_verifiable(self):
        for name in f.ALL_FAMILIES:
            for item in self.data[name+'_test'][:5]:
                self.assertIn(item.hidden_graph,g.GRAPHS)
                self.assertTrue(t.certificate(item,item.hidden_graph)['accepted'])
                self.assertTrue(t.certificate(item,item.hidden_graph)['future_correct'])


class ModelsAndAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoints,cls.historical,cls.digests=b.checkpoints()
        cls.classes=l.class_map()
        cls.sample=f.roster(include_test=False,train_per=2,dev_per=2)['dev'][0]

    def test_checkpoint_source_locks_and_matched_capacity(self):
        self.assertEqual(set(self.checkpoints),set(tr.ROLES))
        self.assertEqual(len(self.digests),len(tr.ROLES)+2)
        self.assertEqual(self.checkpoints['balanced_recurrent']['parameter_count'],6240)
        self.assertEqual(self.checkpoints['balanced_feedforward']['parameter_count'],6235)
        for k,v in self.checkpoints.items():
            self.assertEqual(v['source_sha256'],tr.source_hashes())
            self.assertEqual(v['optimizer_steps'],tr.STEPS)
            self.assertFalse(v['future_access_at_inference'])
            self.assertEqual(v['neural_candidate_previews'],0)
            self.assertEqual(v['train_roster_sha256'],f.digest(f.roster(include_test=False)['train']))

    def test_balanced_vs_skewed_sampling_budget(self):
        a=self.checkpoints['balanced_recurrent']
        z=self.checkpoints['skewed_recurrent']
        self.assertEqual(a['parameter_count'],z['parameter_count'])
        self.assertEqual(a['optimizer_steps'],z['optimizer_steps'])
        self.assertEqual(a['train_roster_sha256'],z['train_roster_sha256'])
        self.assertEqual(sum(a['sample_counts'].values()),tr.STEPS*tr.BATCH)
        self.assertEqual(sum(z['sample_counts'].values()),tr.STEPS*tr.BATCH)
        self.assertLess(max(a['sample_counts'].values())-min(a['sample_counts'].values()),1200)
        self.assertGreater(z['sample_counts']['signed']/(tr.STEPS*tr.BATCH),.75)

    def test_zero_unverified_graph_execution_in_student(self):
        case=self.sample
        with patch.object(t,'execute',side_effect=AssertionError('candidate interpreter used before proposal')):
            for ck in self.checkpoints.values():
                self.assertEqual(len(m.logits(case,ck)),64)

    def test_paid_verification_calls_and_work_proxy(self):
        case=self.sample
        neural=m.logits(case,self.checkpoints['balanced_recurrent'])
        order=b.hybrid_order(neural,case.topology,self.classes)
        self.assertEqual(len(order),len(set(order)))
        self.assertEqual(len(order),12)
        with patch.object(t,'execute',wraps=t.execute) as execute:
            result=l.search(case,order,budget=f.BUDGET)
        self.assertEqual(execute.call_count,2*len(result['trace']))
        self.assertEqual(result['integer_loop_proxy'],sum(2*t.graph_cost(p,case.topology) for p in result['trace']))

    def test_portable_inference_matches_torch_if_present(self):
        try:
            import torch
        except ImportError:self.skipTest('CPU torch optional')
        torch.set_num_threads(1)
        for role in ('balanced_recurrent','balanced_feedforward'):
            ck=self.checkpoints[role]
            net=tr.train18.modules()[ck['kind']]()
            net.load_state_dict({n:torch.tensor(x) for n,x in ck['state_dict'].items()})
            x=l.flat(self.sample) if ck['kind']=='feedforward' else l.sequence(self.sample)
            with torch.no_grad():pred=net(torch.tensor([x],dtype=torch.float32))[0].tolist()
            other=m.logits(self.sample,ck)
            self.assertLess(max(abs(a-z) for a,z in zip(pred,other)),5e-5)

    def test_model_predictions_independent_of_future_and_graph(self):
        altered=replace(self.sample,future=(1,)*33,hidden_graph=None,identifier='tampered')
        for ck in self.checkpoints.values():
            self.assertEqual(m.logits(self.sample,ck),m.logits(altered,ck))
        scores=m.logits(self.sample,self.checkpoints['balanced_recurrent'])
        out=b.propose(self.sample,'balanced_gru12',{'balanced_recurrent':scores},self.classes)
        out2=b.propose(altered,'balanced_gru12',{'balanced_recurrent':scores},self.classes)
        self.assertEqual(out,out2)

    def test_frozen_receipt_and_exact_boundaries(self):
        obj=json.loads((HERE/'results'/'rrc19_pilot.json').read_text())
        self.assertEqual(obj['source_sha256'],b.source_hashes())
        self.assertEqual(obj['checkpoint_sha256'],self.digests)
        self.assertEqual(obj['parameters']['budget'],12)
        self.assertEqual(obj['scores']['product_test']['balanced_gru12']['correct_future'],134)
        self.assertEqual(obj['scores']['rational_test']['balanced_gru12']['correct_future'],99)
        self.assertEqual(obj['scores']['squared_test']['skewed_gru12']['correct_future'],125)
        self.assertEqual(obj['scores']['negative']['complete_classes']['accepted'],0)
        self.assertEqual(obj['scores']['collision']['complete_classes']['false_accept'],f.COLLISION_PAIRS)
        self.assertTrue(all(row['same_proposals'] for row in obj['collision_pairs'].values()))

    def test_replication_roster_disjoint_and_success_ceiling(self):
        receipt=json.loads((HERE/'results'/'rrc19_pilot.json').read_text())
        seeds=receipt['posthoc_independent_seed_checks']
        self.assertEqual(len(seeds),2)
        for seed,groups in seeds.items():
            self.assertEqual(set(groups),{'product','rational','squared'})
            for fam,info in groups.items():
                self.assertEqual(info['scores']['complete_classes']['accepted'],80)
                self.assertEqual(info['scores']['balanced_gru12']['cases'],80)

    def test_source_scope_does_not_claim_new_operators(self):
        report=json.loads((HERE/'results'/'rrc19_pilot.json').read_text())
        self.assertIn('no new primitive',report['scope'])
        self.assertEqual(report['parameters']['offline_symbolic_polynomial_normalizations'],256)


if __name__=='__main__':unittest.main()
