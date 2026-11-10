"""RRC-24 exact delay algebra, information boundaries, models, and receipts."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

try:
    from . import delay24 as d, model24 as m, train24 as tr, bench24 as b
except ImportError:
    import delay24 as d
    import model24 as m
    import train24 as tr
    import bench24 as b

HERE=Path(__file__).resolve().parent

class ExactAlgebra(unittest.TestCase):
    def test_first_residual_coefficient_identifies_delayed_factor(self):
        f=(1,2,-1,3)+(0,)*80
        for base_op in d.OPS:
            base=d.operate(f,base_op)
            for k in (16,32,64):
                for c in d.AMPLITUDES:
                    out=d.apply_factor(base,k,c)
                    self.assertEqual(d.first_difference_degree(out,base),k)
                    self.assertEqual(out[k]-base[k],c)

    def test_all_32_factors_share_prefix_through_63(self):
        f=(1,2,3,4)+(0,)*190
        for op in d.OPS:
            outcomes=[d.execute(f,(op,k,c)) for k in d.DELAYS for c in d.AMPLITUDES]
            self.assertEqual(len(outcomes),32)
            self.assertEqual(len({x[:64] for x in outcomes}),1)
            self.assertEqual(len({tuple(x) for x in outcomes}),32)

    def test_exact_factorized_recovery_per_base(self):
        for i,op in enumerate(d.OPS):
            rng=random.Random(1000+i)
            case=d._sample_case(rng,'signed',f'a-{i}',op=op)
            item=d.verify_candidate(case,op)
            self.assertEqual(item['program'],case.program)
            self.assertEqual(d.grade(case,item['program'])['future_correct'],1)

    def test_delays_outside_declared_grammar_rejected(self):
        rng=random.Random(812)
        c=d._sample_case(rng,'signed','invalid-delay',delay=65,amplitude=1,op='I')
        self.assertIsNone(d.verify_candidate(c,'I')['program'])
        self.assertIsNone(d.search(c,d.OPS)['program'])

    def test_two_demonstrations_must_agree_on_factor(self):
        rng=random.Random(813)
        c=d._sample_case(rng,'signed','mixed',delay=64,amplitude=2,op='I')
        a,_=c.demos[1]
        new=d.execute(a,('I',80,2))
        mutant=replace(c,demos=(c.demos[0],(a,new)))
        self.assertIsNone(d.verify_candidate(mutant,'I')['program'])

    def test_model_prefix_cannot_identify_delayed_factor(self):
        rng=random.Random(814)
        f=d.generate_source(rng,'signed')
        h=d.generate_source(rng,'signed')
        variants=[]
        for k in d.DELAYS:
            for amp in d.AMPLITUDES:
                c=d.Case('same','signed',((f,d.execute(f,('N',k,amp))),
                            (h,d.execute(h,('N',k,amp)))),f,f,('N',k,amp))
                variants.append(d.learn_features(c))
        self.assertEqual(len(variants),32)
        self.assertEqual(len(set(variants)),1)

    def test_invalid_and_inconsistent_series_rejected(self):
        with self.assertRaises(ValueError):d.operate((2,1),'I')
        with self.assertRaises(ValueError):d.execute((1,1,0),('I',3,1))
        with self.assertRaises(ValueError):d.apply_factor((1,0,0),None,1)
        with self.assertRaises(ValueError):d.search(None,d.OPS,0)
        with self.assertRaises(AttributeError):d.learn_features(object())

    def test_future_collision_information_limit(self):
        cases=d.dataset(train_per=2,dev_per=2,test_per=3)['collision']
        self.assertEqual(len(cases),16)
        for base,mutant in zip(cases[::2],cases[1::2]):
            self.assertEqual(base.visible,mutant.visible)
            self.assertEqual(base.query,mutant.query)
            self.assertNotEqual(base.future,mutant.future)
            self.assertEqual(d.first_difference_degree(base.future,mutant.future),224)
            result=d.search(base,d.OPS)
            self.assertEqual(result,d.search(mutant,d.OPS))
            self.assertEqual(sum(d.grade(x,result['program'])['future_correct'] for x in (base,mutant)),1)

    def test_negative_exact_rejection(self):
        groups=d.dataset(train_per=2,dev_per=2,test_per=3)
        self.assertTrue(all(d.search(c,d.OPS)['program'] is None for c in groups['negative']))

class DataAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=d.dataset(train_per=3,dev_per=2,test_per=4)
        cls.models,cls.digests=b.load_models()

    def test_training_never_generates_unseen_source_families(self):
        original=d.generate_source
        def deny(rng,family,*args,**kwargs):
            if family in ('rational','squared','late_source'):
                raise AssertionError('heldout generator called during fitting')
            return original(rng,family,*args,**kwargs)
        with patch.object(d,'generate_source',side_effect=deny):
            x=d.dataset(include_test=False,train_per=3,dev_per=2)
        self.assertEqual(set(x),{'train','dev'})
        self.assertEqual(len(x['train']),9)

    def test_model_input_excludes_hidden_program_and_future(self):
        c=self.data['signed'][0]
        hidden=replace(c,program=None,future=(1,)*len(c.future),identifier='tampered')
        self.assertEqual(d.learn_features(c),d.learn_features(hidden))
        for ck in self.models.values():self.assertEqual(m.logits(c,ck),m.logits(hidden,ck))

    def test_dimensions_and_training_locks(self):
        self.assertEqual(m.params('recurrent'),609)
        self.assertEqual(m.params('feedforward'),588)
        for name,obj in self.models.items():
            self.assertEqual(obj['kind'],name)
            self.assertEqual(obj['parameter_count'],m.params(name))
            self.assertEqual(obj['source_sha256'],tr.source_hashes())
            self.assertEqual(obj['target'],'base operator identity only, not delay or amplitude')

    def test_torch_and_portable_logits_parity(self):
        try:
            import torch
            from torch import nn
        except ImportError:self.skipTest('torch optional')
        torch.set_num_threads(1)
        c=self.data['product'][0]
        seq=torch.tensor([d.learn_features(c)],dtype=torch.float32)
        for kind in ('recurrent','feedforward'):
            ck=self.models[kind]
            if kind=='recurrent':
                recurrent=nn.GRU(4,m.HIDDEN,batch_first=True)
                head=nn.Linear(m.HIDDEN,4)
                recurrent.load_state_dict({k[4:]:torch.tensor(v) for k,v in ck['state_dict'].items() if k.startswith('rnn.')})
                head.load_state_dict({k[5:]:torch.tensor(v) for k,v in ck['state_dict'].items() if k.startswith('head.')})
                with torch.no_grad():actual=head(recurrent(seq)[0][:,-1])[0].tolist()
            else:
                ff=nn.Linear(68,m.FF_HIDDEN);head=nn.Linear(m.FF_HIDDEN,4)
                ff.load_state_dict({k[3:]:torch.tensor(v) for k,v in ck['state_dict'].items() if k.startswith('fc.')})
                head.load_state_dict({k[5:]:torch.tensor(v) for k,v in ck['state_dict'].items() if k.startswith('head.')})
                with torch.no_grad():actual=head(torch.tanh(ff(seq.flatten(1))))[0].tolist()
            portable=m.logits(c,ck)
            self.assertLess(max(abs(x-y) for x,y in zip(actual,portable)),2e-5)

    def test_short_prefix_classifier_explicitly_executes_primitives(self):
        c=self.data['signed'][0]
        original=d.operate
        with patch.object(d,'operate',wraps=original) as called:
            ranking=d.base_order_exact_prefix(c)
        self.assertEqual(called.call_count,8)
        self.assertEqual(ranking[0],c.program[0])

    def test_late_source_neural_inputs_are_all_identical(self):
        cases=self.data['late_source']
        self.assertEqual(len({d.learn_features(c) for c in cases}),1)
        self.assertEqual(len({m.order(c,self.models['recurrent']) for c in cases}),1)

    def test_actual_full_verifier_call_limit(self):
        c=self.data['squared'][0]
        original=d.operate
        with patch.object(d,'operate',wraps=original) as f:
            z=b.predict(c,'recurrent_2',self.models)
        self.assertEqual(f.call_count,2*z['queries'])
        self.assertLessEqual(z['queries'],2)
        self.assertEqual(z['base_executions'],2*z['queries'])
        self.assertEqual(len(set(z['trace'])),z['queries'])

class BudgetAndReplay(unittest.TestCase):
    def test_literal_program_count_and_progressive_stages(self):
        self.assertEqual(len(b.flat_catalogue(d.OPS)),132)
        self.assertEqual(len(set(b.flat_catalogue(d.OPS))),132)
        case=d.dataset(train_per=2,dev_per=2,test_per=2)['signed'][0]
        out=b.flat_search(case,d.OPS,budget=12)
        self.assertLessEqual(out['queries'],12)
        self.assertEqual(out['demo_stage_executions'],2*sum(out['stage_counts'].values()))
        self.assertGreater(out['nominal_work'],0)

    def test_no_future_sensitive_search_order(self):
        model,_=b.load_models();case=d.dataset(train_per=2,dev_per=2,test_per=2)['squared'][0]
        changed=replace(case,program=None,future=tuple([1]*len(case.future)))
        for arm in b.ARMS:
            self.assertEqual(b.predict(case,arm,model),b.predict(changed,arm,model))

    def test_source_bound_receipt_and_per_family_ceiling(self):
        rec=json.loads((HERE/'results'/'rrc24_pilot.json').read_text())
        self.assertEqual(rec['source_sha256'],b.hashes())
        self.assertEqual(rec['checkpoint_sha256'],b.load_models()[1])
        self.assertEqual(rec['configuration']['model_input_degree'],16)
        self.assertEqual(rec['configuration']['full_syntax_candidates'],132)
        for family in d.TEST_FAMILIES:
            self.assertEqual(rec['scores'][family]['exhaustive_4']['future_correct'],40)
        self.assertEqual(rec['scores']['negative']['exhaustive_4']['accepted'],0)
        self.assertEqual(rec['scores']['collision']['exhaustive_4']['false_accept'],8)
        self.assertEqual(rec['flat_syntax_diagnostic']['132']['accepted'],12)

    def test_physical_profile_independent_of_deterministic_receipt(self):
        data=json.loads((HERE/'results'/'rrc24_reference_profile.json').read_text())
        self.assertTrue(data['physical_times_are_machine_specific'])
        self.assertEqual(data['cases'],36)
        self.assertEqual(data['rounds'],2)
        for x in data['timings'].values():
            self.assertGreater(x['median_total_ms'],0)
            self.assertEqual(x['evaluations'],72)

if __name__=='__main__':unittest.main()
