"""RRC-09: independent exact arithmetic, separation and reproducibility tests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest import mock

try:
    from . import synth09 as s, model09, bench09
    from .rrc import gap_coefficients
    from .rrc05 import Program05,product_coefficients
except ImportError:
    import synth09 as s
    import model09
    import bench09
    from rrc import gap_coefficients
    from rrc05 import Program05,product_coefficients

HERE=Path(__file__).resolve().parent


def independent_product(degree,parts,cap):
    """Brute enumeration, independent of source's polynomial DP."""
    def rec(n,j):
        if j==len(parts):return int(n==0)
        allowed=n//parts[j] if cap==0 else min(cap,n//parts[j])
        return sum(rec(n-k*parts[j],j+1) for k in range(allowed+1))
    return [rec(n,0) for n in range(degree+1)]


class AlgebraTests(unittest.TestCase):
    def test_public_primitive_semantics(self):
        f=(1,1,1,1,1,1,1)
        self.assertEqual(s.operate(f,'R'),(1,-1,0,0,0,0,0))
        self.assertEqual(s.operate(f,'N'),(1,-1,1,-1,1,-1,1))
        self.assertEqual(s.operate(f,'S'),(1,0,1,0,1,0,1))
        self.assertEqual(s.operate(f,'D'),(1,1,0,0,0,0,0))
        self.assertEqual(s.operate((1,1,0,0),'C'),(1,2,1,0))
        with self.assertRaises(ValueError):s.operate(f,'BAD')

    def test_brute_partitions_with_caps(self):
        for cap in (0,1,2):
            for mask in ((1,4),(2,3)):
                f=product_coefficients(12,Program05(cap,5,mask))
                parts=[i for i in range(1,13) if i%5 in mask]
                self.assertEqual(f,independent_product(12,parts,cap))

    def test_rogers_ramanujan_euler_inversion(self):
        for minimum,residues in ((1,(1,4)),(2,(2,3))):
            f=gap_coefficients(25,minimum_part=minimum)
            e=s.operate(f,'D')
            self.assertEqual(e[0],1)
            self.assertEqual(list(e[1:]),[int(k%5 in residues) for k in range(1,26)])

    def test_composition_is_executable_and_not_commutative(self):
        f=(1,1,1,1,1,1,1,1,1)
        self.assertEqual(s.execute(f,('N','N')),f)
        self.assertEqual(s.execute(f,('R','R')),f)
        self.assertEqual(s.execute(f,('C','R')),s.execute(f,('R','C')))
        x=s.execute(f,('D','C'));y=s.execute(f,('C','D'))
        self.assertNotEqual(x,y)
        self.assertEqual(len(x),len(f))
        self.assertEqual(len(s.all_programs()),31)
        self.assertEqual(len(set(s.all_programs())),31)

    def test_finite_prefix_synthesis_and_bounded_cost(self):
        r=s.generate(include_test=True,train_count=10,dev_count=5)
        c=r['heldout_CD'][0]
        found=s.catalog_search(c.source,c.target)
        self.assertIsNotNone(found['program'])
        self.assertEqual(s.execute(c.source[:s.PREFIX+1],found['program']),c.target[:s.PREFIX+1])
        self.assertLessEqual(found['candidates_examined'],31)
        self.assertGreater(found['estimated_arithmetic_work'],0)

    def test_unrecognized_series_rejected(self):
        cases=s.generate(include_test=True,train_count=12,dev_count=8)['out_of_grammar']
        self.assertEqual(len(cases),24)
        self.assertTrue(all(s.catalog_search(c.source,c.target)['program'] is None for c in cases))

    def test_validation_and_source_contract(self):
        with self.assertRaises(ValueError):s.valid_series((2,3))
        with self.assertRaises(ValueError):s.execute((1,1.3),('N',))
        with self.assertRaises(ValueError):s.catalog_search((1,2,3),(1,2,3))
        with self.assertRaises(ValueError):s.guided_search((1,2),(1,2),[1]*6,budget=0)
        with self.assertRaises(ValueError):s.arithmetic_work(('BAD',),6)


class FamilyAndInformationTests(unittest.TestCase):
    def test_no_heldout_primitives_in_training_generator(self):
        with mock.patch.object(s,'_target_programs',side_effect=AssertionError('test labels constructed')):
            d=s.generate(include_test=False,train_count=24,dev_count=12)
        self.assertEqual(len(d['train']),24)
        self.assertEqual(len(d['dev']),12)
        self.assertTrue(all(set(c.hidden_program).isdisjoint({'C','D'})
                            for split in d.values() for c in split))

    def test_disjoint_family_splits_and_explicit_pairs(self):
        d=s.generate(train_count=40,dev_count=10)
        self.assertEqual({k:len(v) for k,v in d.items() if k!='train' and k!='dev'},s.TEST_COUNTS)
        train={c.input_signature for c in d['train']}
        for name,cases in d.items():
            if name=='train':continue
            self.assertFalse(train.intersection({c.input_signature for c in cases}))
        for a,b in zip(d['late_collision'][::2],d['late_collision'][1::2]):
            self.assertEqual(a.input_signature,b.input_signature)
            self.assertEqual(s.first_fault(a.target,b.target),s.first_fault(a.target,b.target,s.PREFIX+1))
            self.assertIsNotNone(s.first_fault(a.target,b.target,s.PREFIX+1))

    def test_both_collision_members_induce_same_program(self):
        cases=s.generate(train_count=20,dev_count=10)['late_collision']
        models=bench09.load_checkpoints()
        for a,b in zip(cases[::2],cases[1::2]):
            p=s.catalog_search(a.source,a.target)['program']
            self.assertEqual(p,s.catalog_search(b.source,b.target)['program'])
            self.assertEqual(model09.direct_program(a,models['recurrent'][0]),
                             model09.direct_program(b,models['recurrent'][0]))
            self.assertEqual(s.guided_search(a.source,a.target,models['guided_ranker'][0]['weights'],budget=4)['program'],
                             s.guided_search(b.source,b.target,models['guided_ranker'][0]['weights'],budget=4)['program'])
            if p is not None:
                successes=sum(bench09.verify_program(case,p)['correct_future'] for case in (a,b))
                self.assertEqual(successes,1)

    def test_candidate_ranking_requires_interpreter_while_direct_does_not(self):
        r=s.generate(train_count=20,dev_count=10)
        c=r['heldout_CD'][0];model=bench09.load_checkpoints()['recurrent'][0]
        with mock.patch.object(s,'execute',side_effect=AssertionError('direct model accessed interpreter')):
            direct=model09.direct_program(c,model)
        self.assertLessEqual(len(direct),2)
        with self.assertRaises(AssertionError):
            with mock.patch.object(s,'execute',side_effect=AssertionError('tool interpreter used')):
                s.guided_search(c.source,c.target,None,budget=1)

    def test_future_mutations_cannot_influence_prefix_proposals(self):
        c=s.generate(train_count=20,dev_count=10)['heldout_D'][0]
        other=replace(c,target=c.target[:s.PREFIX+1]+tuple(-v for v in c.target[s.PREFIX+1:]))
        ck=bench09.load_checkpoints()
        self.assertEqual(c.input_signature,other.input_signature)
        self.assertEqual(s.catalog_search(c.source,c.target),s.catalog_search(other.source,other.target))
        self.assertEqual(model09.direct_program(c,ck['feedforward'][0]),
                         model09.direct_program(other,ck['feedforward'][0]))
        self.assertEqual(s.guided_search(c.source,c.target,ck['guided_ranker'][0]['weights']),
                         s.guided_search(other.source,other.target,ck['guided_ranker'][0]['weights']))


class ModelAndReplayTests(unittest.TestCase):
    def test_checkpoint_locks_and_budget(self):
        ck=bench09.load_checkpoints()
        self.assertEqual(ck['recurrent'][0]['parameter_count'],1692)
        self.assertEqual(ck['feedforward'][0]['parameter_count'],1695)
        self.assertEqual(ck['guided_ranker'][0]['train_examples'],900)
        self.assertEqual(ck['guided_ranker'][0]['trained_candidate_ops'],['N','S','R'])
        self.assertTrue(all(c['train_excludes_test_generators']
                            for key,(c,_) in ck.items() if key!='guided_ranker'))

    def test_direct_grammar_predictions_always_verifier_checked(self):
        r=s.generate(train_count=12,dev_count=5)
        c=r['heldout_C'][0]
        ck=bench09.load_checkpoints()
        for key in ('recurrent','feedforward','shuffled_recurrent'):
            p=model09.direct_program(c,ck[key][0])
            self.assertTrue(all(op in s.OPS for op in p))
            accepted=bench09.verify_program(c,p)
            self.assertEqual(accepted['prefix_verified'],int(s.execute(c.source[:s.PREFIX+1],p)==c.target[:s.PREFIX+1]))

    def test_exact_search_and_preview_guards(self):
        c=s.generate(train_count=12,dev_count=5)['heldout_D'][0]
        ck=bench09.load_checkpoints()['guided_ranker'][0]
        for w in (ck['weights'],None):
            for budget in (1,4,31):
                p=s.guided_search(c.source,c.target,w,budget=budget)
                self.assertEqual(p['preview_candidates'],31)
                self.assertLessEqual(p['full_verifications'],budget)
                self.assertGreater(p['primitive_calls'],0)

    def test_torch_vs_portable_inference_if_available(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest('PyTorch optional; CI inference is standard-library')
        torch.set_num_threads(1)
        c=s.generate(train_count=12,dev_count=5)['dev'][0]
        ck=bench09.load_checkpoints()
        x=torch.tensor([s.feature_pairs(c)],dtype=torch.float32)
        for key in ('recurrent','feedforward'):
            content=ck[key][0]
            if key=='recurrent':
                net=nn.GRU(input_size=2,hidden_size=20,batch_first=True)
                head=nn.Linear(20,12)
                for part,mod in (('rnn',net),('head',head)):
                    state={name:torch.tensor(value) for name,value in content['state_dict'].items()
                           if name.startswith(part+'.')}
                    mod.load_state_dict({name[len(part)+1:]:v for name,v in state.items()})
                with torch.no_grad():actual=head(net(x)[0][:,-1])[0].tolist()
            else:
                first=nn.Linear(38,33);head=nn.Linear(33,12)
                first.load_state_dict({name[4:]:torch.tensor(v) for name,v in content['state_dict'].items() if name.startswith('fc1.')})
                head.load_state_dict({name[5:]:torch.tensor(v) for name,v in content['state_dict'].items() if name.startswith('head.')})
                with torch.no_grad():actual=head(torch.tanh(first(x.reshape(1,-1))))[0].tolist()
            portable=model09.direct_logits(c,content)
            self.assertLess(max(abs(a-b) for a,b in zip(actual,portable)),2e-5)

    def test_full_heldout_receipt_contract(self):
        data=json.loads((HERE/'results'/'rrc09_pilot.json').read_text())
        self.assertEqual(data['parameters']['test_counts'],s.TEST_COUNTS)
        for name in ('heldout_C','heldout_D','heldout_CD'):
            count=s.TEST_COUNTS[name]
            self.assertEqual(data['metrics']['exact']['by_split'][name]['prefix_verified'],count)
            self.assertEqual(data['metrics']['guided_4']['by_split'][name]['prefix_verified'],count)
            self.assertEqual(data['metrics']['recurrent_1']['by_split'][name]['prefix_verified'],0)
            self.assertEqual(data['metrics']['feedforward']['by_split'][name]['prefix_verified'],0)
        self.assertEqual(data['metrics']['exact']['by_split']['late_collision']['false_accept'],8)
        self.assertEqual(data['metrics']['exact']['by_split']['out_of_grammar']['abstentions'],24)


if __name__=='__main__':unittest.main()
