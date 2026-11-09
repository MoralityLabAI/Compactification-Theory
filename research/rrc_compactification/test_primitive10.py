"""RRC-10 independent algebra, separation, and leakage tests."""
from __future__ import annotations
import unittest
from unittest import mock
try:
    from . import primitive10 as p,train10,bench10
except ImportError:
    import primitive10 as p
    import train10,bench10

class Algebra(unittest.TestCase):
    def test_convolution(self):
        self.assertEqual(p.mul((1,1,0,0),(1,2,1,0)),(1,3,3,1))
    def test_inverse(self):
        for f in ((1,1,0,0,0),(1,-2,1,0,0),(1,0,0,3,1)):
            self.assertEqual(p.mul(f,p.invert(f)),(1,0,0,0,0))
    def test_wrapper(self):
        f=(1,1,2,3,4)
        self.assertEqual(p.apply_wrapper(f,'N'),(1,-1,2,-3,4))
        self.assertEqual(p.apply_wrapper(f,'S'),(1,0,1,0,2))
        self.assertEqual(p.apply_wrapper(p.apply_wrapper(f,'R'),'R'),f)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):p.multiplier(0,'single')
        with self.assertRaises(ValueError):p.multiplier(25,'double')
        with self.assertRaises(ValueError):p.apply_wrapper((1,),'Q')
        with self.assertRaises(ValueError):p.invert((0,1))
        with self.assertRaises(ValueError):p.recovery(((1,1),(1,1)),(1,),1)
    def test_exact_teacher_forced_inversion(self):
        d=p.episodes(include_test=False)['train']
        for c in d[:25]:
            k=p.recovery(c.demos[0],(1,-1))
            self.assertEqual(p.mul(c.demos[0][0][:p.DEMO+1],k),c.demos[0][1][:p.DEMO+1])

class Protocol(unittest.TestCase):
    def test_heldout_k_and_types(self):
        r=p.episodes()
        self.assertEqual(len(r['train']),128)
        self.assertEqual(len(r['new_multiplier']),48)
        self.assertTrue(all(c.k in p.TRAIN_K and c.kind=='single' for c in r['train']))
        self.assertTrue(all(c.k in p.TEST_K and c.kind!='single' for c in r['new_multiplier']))
    def test_no_test_data_created_by_training(self):
        with mock.patch.object(p,'multiplier',wraps=p.multiplier) as fn:
            d=p.episodes(include_test=False)
            self.assertEqual(set(d),{'train','dev'})
            self.assertTrue(all(call.args[1]=='single' for call in fn.call_args_list))
    def test_inference_does_not_see_hidden_template(self):
        e=p.episodes()['new_multiplier'][0]
        with mock.patch.object(p,'multiplier',side_effect=AssertionError('hidden constructor leaked')):
            self.assertTrue(p.propose(e,(1,-1))['accepted'])
    def test_out_of_range_future_unidentifiable(self):
        pair=p.episodes()['late_collision']
        for a,b in zip(pair[::2],pair[1::2]):
            self.assertEqual(a.demos,b.demos)
            self.assertEqual(a.query,b.query)
            self.assertNotEqual(a.future,b.future)
            self.assertEqual(p.propose(a,(1,-1)),p.propose(b,(1,-1)))
    def test_conflicts_rejected(self):
        for c in p.episodes()['conflict']:
            self.assertEqual(p.propose(c,(1,-1))['reason'],'demo_disagreement')
    def test_learning_is_from_training_only(self):
        r=train10.fit()
        self.assertEqual(r['compiled_integer_weights'],[1,-1])
        self.assertEqual(r['supervised_rows'],128*2*p.DEMO)
    def test_local_baseline_coverage(self):
        r=train10.fit(local=True)
        self.assertEqual(r['kind'],'raw_local')
    def test_recurrent_compiler_heldout(self):
        for group,cases in p.episodes().items():
            if group in ('train','dev'):continue
            for c in cases:
                a=p.propose(c,(1,-1))
                if group=='conflict':self.assertFalse(a['accepted'])
                else:self.assertTrue(a['accepted'])
    def test_receipt(self):
        r=bench10.run()
        self.assertEqual(r['metrics']['recurrent']['splits']['new_composition']['future_correct'],48)
        self.assertEqual(r['metrics']['recurrent']['splits']['late_collision']['future_correct'],8)
        self.assertEqual(r['metrics']['recurrent']['splits']['conflict']['abstention'],16)
        self.assertTrue(all(c['correct_members']==1 for c in r['collision_pairs']))

if __name__=='__main__':unittest.main()
