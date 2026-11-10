"""RRC-25: independent algebra, finite ambiguity, model and audit checks."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
try:
    from . import compound25 as b, bench25 as e, model25 as m, train25 as tr
    from . import delay24 as d
except ImportError:
    import compound25 as b
    import bench25 as e
    import model25 as m
    import train25 as tr
    import delay24 as d
HERE=Path(__file__).resolve().parent


class AlgebraAndIdentification(unittest.TestCase):
    def test_first_two_faults_cannot_identify_branches(self):
        f=(1,2,1)+tuple((i%3)-1 for i in range(b.FUTURE-2))
        p=('I',80,24,2,'I','R')
        a=b.execute(f,p)
        u_alt=b.execute(f,('I',80,24,2,'N','R'))
        v_alt=b.execute(f,('I',80,24,2,'I','N'))
        self.assertEqual(a[:105],u_alt[:105])
        self.assertNotEqual(a[105],u_alt[105])
        self.assertEqual(a[:130],v_alt[:130])
        self.assertNotEqual(a[130],v_alt[130])
        base=d.operate(f,'I')
        self.assertEqual(a[80]-base[80],2)
        self.assertEqual(a[104]-base[104],2)

    def test_explicit_coefficient_recurrence_matches_generator(self):
        rng=random.Random(25)
        for _ in range(10):
            f=(1,)+tuple(rng.randint(-2,2) for _ in range(b.FUTURE))
            program=(rng.choice(b.OPS),rng.choice(b.DELAYS),rng.choice(b.SPACINGS),
                    rng.choice(b.AMPLITUDES),rng.choice(b.OPS),rng.choice(b.OPS))
            op,k,delta,c,u,v=program
            y=b.execute(f,program)
            first=d.operate(f,op);left=d.operate(f,u);right=d.operate(f,v)
            for n in range(b.FUTURE+1):
                self.assertEqual(y[n],first[n]+(c if n==k else 0)
                    +(c*left[n-k-delta] if n>=k+delta else 0)
                    +(c*right[n-k-2*delta] if n>=k+2*delta else 0))

    def test_invalid_public_program_and_degree_contract(self):
        f=(1,)+(0,)*b.FUTURE
        with self.assertRaises(ValueError):b.execute(f,('X',64,24,1,'I','R'))
        with self.assertRaises(ValueError):b.execute(f,('I',64,24,0,'I','R'))
        with self.assertRaises(ValueError):b.execute(f,('I',32,24,1,'I','R'))
        with self.assertRaises(ValueError):b.execute(f,('I',64,24,1,'I','Q'))
        with self.assertRaises(ValueError):b.search(None,'fixed',budget=0)
        with self.assertRaises(ValueError):b.search(None,'fixed',budget=65)
        self.assertEqual(len(b.OPS)*len(b.DELAYS)*len(b.SPACINGS)*len(b.AMPLITUDES)*16,2048)

    def test_header_discovery_uses_only_public_pairs(self):
        c=b.roster(include_test=False,train_per=4,dev_per=2)['dev'][0]
        h=b.possible_headers(c)
        self.assertIn(b.Header(*c.program[:4]),h)
        altered=replace(c,program=None,future=(1,)*len(c.future),identifier='redacted')
        self.assertEqual(b.possible_headers(altered),h)
        self.assertEqual(b.features(c,h[0]),b.features(altered,h[0]))

    def test_preview_and_full_program_checker_agree_on_true_branch(self):
        cases=b.roster(include_test=False,train_per=5,dev_per=3)['dev']
        for c in cases:
            h=b.Header(*c.program[:4]);u,v=c.program[4:]
            self.assertEqual(b.preview_errors(c,h,u,v),(2*(b.WINDOW+1)*2,0))
            self.assertTrue(b.verify_template(c,h,u,v))
            self.assertTrue(b.grade(c,c.program)['future_correct'])


class FamilyAndUncertainty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=b.roster(include_test=True,train_per=10,dev_per=5,test_per=4)

    def test_training_never_constructs_novel_family(self):
        original=d.generate_source
        def guard(rng,family,degree=b.FUTURE):
            if family in ('rational','squared','late_source'):
                raise AssertionError('heldout source generated during training')
            return original(rng,family,degree=degree)
        with patch.object(d,'generate_source',side_effect=guard):
            groups=b.roster(include_test=False,train_per=5,dev_per=3)
        self.assertEqual(set(groups),{'train','dev'})
        self.assertEqual(len(groups['train']),15)
        self.assertTrue(all(z.family in b.TRAIN_FAMILIES for z in groups['train']))

    def test_visible_train_test_disjoint_and_collision_pairs_explicit(self):
        known={c.visible for c in self.cases['train']}
        for key,items in self.cases.items():
            if key=='train':continue
            self.assertFalse(known.intersection({c.visible for c in items}))
        for a,c in zip(self.cases['collision'][::2],self.cases['collision'][1::2]):
            self.assertEqual(a.visible,c.visible)
            self.assertEqual(a.query,c.query)
            self.assertNotEqual(a.future,c.future)

    def test_exhaustive_finite_prefix_reference_accepts_all_positives(self):
        for fam in b.TEST_FAMILIES:
            for c in self.cases[fam][:2]:
                z=b.complete_search(c)
                self.assertTrue(b.grade(c,z['program'])['accepted'])
                self.assertLessEqual(z['full_checks'],64)
                self.assertGreaterEqual(z['header_base_executions'],4)
                self.assertLessEqual(z['header_base_executions'],8)

    def test_negative_cases_outside_declared_grammar(self):
        for c in self.cases['negative']:
            self.assertIsNone(b.complete_search(c)['program'])
            self.assertEqual(b.visible_template_count(c),0)

    def test_verified_program_ambiguity_matches_literal_exhaustion(self):
        for family in ('signed','rational','late_source'):
            for case in self.cases[family][:2]:
                h=b.possible_headers(case)
                exact=sum(int(b.verify_template(case,x,u,v)) for x in h for u in b.OPS for v in b.OPS)
                self.assertEqual(b.visible_template_count(case),exact)

    def test_paid_preview_is_separate_from_full_verification(self):
        c=self.cases['signed'][0]
        exact=b.search(c,'preview',budget=4)
        self.assertEqual(exact['preview_pairs'],16)
        self.assertLessEqual(exact['full_checks'],4)
        self.assertEqual(len(set(exact['trace'])),exact['full_checks'])
        self.assertGreater(b.total_work(exact),b.total_work(b.search(c,'fixed',budget=4)))

    def test_heldout_future_pair_is_identical_to_every_policy(self):
        models,_=e.checkpoints()
        for a,c in zip(self.cases['collision'][::2],self.cases['collision'][1::2]):
            for arm in e.ARMS:
                x=e.predict(a,arm,models);y=e.predict(c,arm,models)
                self.assertEqual((x['trace'],x['program']),(y['trace'],y['program']))
            result=b.complete_search(a)['program']
            self.assertEqual(b.grade(a,result)['future_correct']+b.grade(c,result)['future_correct'],1)


class ModelAndReceipts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.digests=e.checkpoints()
        cls.sample=b.roster(include_test=False,train_per=6,dev_per=3)['dev'][0]

    def test_model_shape_and_source_receipts(self):
        self.assertEqual(m.parameters('recurrent'),1000)
        self.assertEqual(m.parameters('feedforward'),995)
        self.assertEqual(set(self.models),set(e.KINDS))
        for role,ck in self.models.items():
            self.assertEqual(ck['source_sha256'],tr.source_hashes())
            self.assertFalse(ck['test_generators_during_training'])
            self.assertEqual(ck['parameters'],m.parameters(role))

    def test_no_interpreter_candidate_invocation_inside_neural_forward(self):
        case=self.sample;h=b.Header(*case.program[:4])
        with patch.object(b,'verify_template',side_effect=AssertionError('candidate checker accessed')):
            for ck in self.models.values():
                self.assertEqual(len(m.logits(case,h,ck)),16)

    def test_torch_and_portable_model_parity(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is optional in exact-only CI')
        torch.set_num_threads(1)
        h=b.Header(*self.sample.program[:4])
        x=torch.tensor([b.features(self.sample,h)],dtype=torch.float32)
        for role in ('recurrent','feedforward'):
            net=tr.modules()[role]()
            net.load_state_dict({k:torch.tensor(v) for k,v in self.models[role]['state_dict'].items()})
            with torch.no_grad():reference=net(x)[0].tolist()
            portable=m.logits(self.sample,h,self.models[role])
            self.assertLess(max(abs(a-z) for a,z in zip(reference,portable)),3e-5)

    def test_saved_score_and_source_contract(self):
        obj=json.loads((HERE/'results'/'rrc25_pilot.json').read_text())
        self.assertEqual(obj['source_sha256'],e.hashes())
        self.assertEqual(obj['checkpoint_sha256'],self.digests)
        self.assertEqual(obj['parameters']['full_literal_grammar'],2048)
        self.assertEqual(obj['scores']['product']['recurrent_4']['future_correct'],32)
        self.assertEqual(obj['scores']['rational']['recurrent_4']['future_correct'],5)
        self.assertEqual(obj['scores']['late_source']['complete_64']['future_correct'],11)
        self.assertEqual(obj['visible_program_ambiguity']['late_source']['multiple_visible_programs'],28)
        self.assertEqual(obj['scores']['negative']['complete_64']['accepted'],0)
        self.assertEqual(obj['future_pairs']['complete_64']['false_future_acceptances'],b.COLLISION_PAIRS)
        self.assertTrue(all(x['identical_trace_and_proposal'] for x in obj['future_pairs'].values()))


if __name__=='__main__':unittest.main()
