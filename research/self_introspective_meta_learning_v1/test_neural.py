"""SIML-01N neural unit and matched-compute smoke tests.

Torch tests skip in an environment that did not install optional CPU torch.
The recorded scientific run comes from a known torch CPU environment.
"""
import importlib.util
import math
import random
import unittest

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from neural_student import (FEATURES,OPERATIONS,NODES,T1_WORDS,DEV_WORDS,
                                TRAIN_WORDS,split_shear,frozen_pool,make_case,
                                RecurrentStudent,encode_batch,apply_word,TRIPLE_WORDS,triple_transfer_accuracy)
    from neural_curriculum import Config,ARMS,choose_task,run_arm,run_experiment,checksum_model

@unittest.skipUnless(HAS_TORCH,'CPU torch optional dependency not installed')
class NeuralStudentTests(unittest.TestCase):
    def test_pool_and_action_word_seals(self):
        self.assertEqual(len(set(TRAIN_WORDS)),18)
        self.assertTrue(set(TRAIN_WORDS).isdisjoint(T1_WORDS))
        self.assertTrue(set(DEV_WORDS).isdisjoint(T1_WORDS))
        self.assertTrue(set(TRAIN_WORDS).isdisjoint(DEV_WORDS))
        x={s:{z.state for z in frozen_pool('T4',s)} for s in ('train','dev','test')}
        self.assertEqual(sum(map(len,x.values())),961)
        self.assertFalse(x['train']&x['dev'] or x['dev']&x['test'] or x['train']&x['test'])
        self.assertTrue(all(split_shear(*z)==s for s in x for z in x[s]))
        self.assertEqual({s:len(frozen_pool('T1',s)) for s in x},
                         {'train':144,'dev':40,'test':16})

    def test_task_targets_are_exact(self):
        for e in frozen_pool('T1','test'):
            self.assertEqual(e.target,apply_word(e.state,e.actions)[0])
        for e in frozen_pool('T4','test'):
            a,b=e.state
            self.assertEqual(e.target,(2*a+b)%31)

    def test_triple_words_are_extra_horizon(self):
        self.assertEqual({len(w) for w in TRIPLE_WORDS},{3})
        self.assertTrue(all(len(w)==2 for w in TRAIN_WORDS))
        torch.manual_seed(91)
        m=RecurrentStudent(16)
        self.assertTrue(0 <= triple_transfer_accuracy(m) <= 1)

    def test_tensor_input_phase_encoding(self):
        x=[make_case('T0',(1,0,1)),make_case('T1',(1,0,1),('flip_b','reveal_b')),
           make_case('T4',(23,20))]
        t=encode_batch(x)
        self.assertEqual(tuple(t[0].shape),(3,FEATURES))
        self.assertEqual(tuple(t[1].shape),(3,2))
        self.assertEqual(t[3].shape,(3,2,3))
        self.assertEqual(t[4].tolist(),[False,True,False])
        self.assertAlmostEqual(t[0][2,67].item(),math.cos(2*math.pi*12/31),places=6)
        self.assertEqual(t[0][2,3+12].item(),1.)

    def test_neural_forward_and_gradient(self):
        torch.manual_seed(71)
        m=RecurrentStudent(16)
        batch=encode_batch([make_case('T0',(1,0,1)),make_case('T1',(0,1,0),('flip_b','flip_y')),
                            make_case('T4',(8,9))])
        logits,states=m(batch[0],batch[1])
        self.assertEqual(tuple(logits.shape),(3,31))
        self.assertEqual(tuple(states.shape),(3,2,16))
        loss=m.batch_loss(batch,successor_weight=.35)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(m.step.weight_ih.grad.norm()),0)
        self.assertGreater(float(m.encoder[0].weight.grad.norm()),0)

    def test_closure_diagnostic_and_repeatability(self):
        torch.manual_seed(13)
        m=RecurrentStudent(16)
        before=checksum_model(m)
        b=frozen_pool('T1','dev')
        self.assertTrue(math.isfinite(m.closure_residual(b)))
        self.assertGreater(m.closure_residual(b),0)
        self.assertEqual(before,checksum_model(m))

    def test_config_validation(self):
        for kwargs in (dict(steps=0),dict(batch=0),dict(seeds=(0,0)),dict(width=4),
                       dict(arms=('bad',)),dict(successor_weight=-1)):
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_policy_uses_hidden_state_measurement(self):
        metrics={'accuracy':{'T0':1.,'T1':.7,'T4':.2},
                 'closure_residual':{'T0':0.,'T1':.5,'T4':.8}}
        for arm in ARMS:
            node=choose_task(arm,0,random.Random(99),metrics,None)
            self.assertIn(node,NODES)
        self.assertEqual(choose_task('introspective_naive',0,random.Random(1),metrics,None),'T4')
        # Guarded scheduler has positive probability of every family.
        choices=[choose_task('introspective',i,random.Random(i),metrics,None)
                 for i in range(250)]
        self.assertEqual(set(choices),set(NODES))

    def test_deterministic_training_with_fixed_seed(self):
        config=Config(seeds=(3,),steps=12,batch=12,width=16,diagnostic_interval=4,
                      arms=('fixed',))
        a=run_arm(config,3,'fixed')
        b=run_arm(config,3,'fixed')
        self.assertEqual(a['trained_checkpoint'],b['trained_checkpoint'])
        self.assertEqual(a['final_test'],b['final_test'])
        self.assertEqual(a['updates_by_node'],b['updates_by_node'])

    def test_matched_update_count_and_model_initialization(self):
        config=Config(seeds=(8,),steps=12,batch=12,width=16,diagnostic_interval=4,
                      arms=ARMS)
        outcomes=run_experiment(config)['outcomes']
        self.assertEqual(len(outcomes),len(ARMS))
        self.assertEqual(len({x['initial_checkpoint'] for x in outcomes}),1)
        self.assertEqual(len({x['resources']['parameters'] for x in outcomes}),1)
        self.assertEqual(len({x['resources']['recurrent_step_examples'] for x in outcomes}),1)
        self.assertEqual(len({x['resources']['diagnostic_rounds'] for x in outcomes}),1)
        for x in outcomes:
            self.assertEqual(sum(x['updates_by_node'].values()),12)
            self.assertEqual(x['resources']['training_updates'],12)
            self.assertTrue(0<=x['weighted_test_score']<=1)

if __name__=='__main__':
    unittest.main()
