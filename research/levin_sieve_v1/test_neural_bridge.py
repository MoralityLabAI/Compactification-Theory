"""LSV-04 exact proof-contract checks (no torch dependency)."""
import copy
import json
import unittest
from fractions import Fraction as Q
from pathlib import Path

from neural_bridge import (TARGET, CLEAN_LIMIT, DAMAGE_LIMIT, WEIGHTS,
    audited_rollout, ex_ante_bound, forcing, guaranteed_absolute_goal,
    load_checkpoint, make_certificate, model_matrix, clip, rollout)
from benchmark_neural import evaluate, roster
from small_gain import Certificate

HERE=Path(__file__).resolve().parent
CHECKPOINT=HERE/'checkpoints/repair_two_node_20261104.json'


class NeuralRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint=load_checkpoint(CHECKPOINT)
        cls.W=model_matrix(cls.checkpoint)
        cls.cert,cls.eta=make_certificate(cls.W)

    def test_checkpoint_fingerprint_and_exact_rational_weights(self):
        self.assertEqual(self.W,((Q(0),Q(433,250)),(Q(369,2000),Q(0))))
        self.assertEqual(self.cert.rho,Q(433,750))
        self.assertEqual(self.cert.radius(self.eta),Q(153,3170))
        self.assertEqual(self.eta,(Q(153,2500),Q(279,20000)))

    def test_tampering_and_bad_model_contract_rejected(self):
        for key,val in [('a01','1.8000'),('a10','0.2000')]:
            ck=copy.deepcopy(self.checkpoint)
            ck['weights'][key]=val
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                model_matrix(ck)
        ck=copy.deepcopy(self.checkpoint)
        ck['target_gains'][0][1]='3'
        with self.assertRaisesRegex(ValueError,'contract changed'):
            model_matrix(ck)

    def test_hard_tanh_global_lipschitz(self):
        grid=[Q(v,2) for v in range(-8,9)]
        for x in grid:
            for y in grid:
                self.assertLessEqual(abs(clip(x)-clip(y)),abs(x-y))
        self.assertEqual(clip(Q(100)),1)
        self.assertEqual(clip(Q(-100)),-1)

    def test_task_goal_exact_fixed_point_for_generator(self):
        for x in (Q(-9,10),Q(0),Q(9,10)):
            for y in (Q(-9,10),Q(0),Q(9,10)):
                goal=(x,y)
                b=forcing(goal)
                for i in range(2):
                    self.assertEqual(goal[i],b[i]+sum(
                        (TARGET[i][j]*clip(goal[j]) for j in range(2)),Q(0)))
        with self.assertRaises(ValueError):
            forcing((Q(1),Q(0)))

    def test_repair_contract_on_adversarial_corners(self):
        for x in (-CLEAN_LIMIT,CLEAN_LIMIT):
            for y in (-CLEAN_LIMIT,CLEAN_LIMIT):
                goal=(x,y)
                for d0 in (-DAMAGE_LIMIT,DAMAGE_LIMIT):
                    for d1 in (-DAMAGE_LIMIT,DAMAGE_LIMIT):
                        for D in (0,1,2,3):
                            states,audit,_,_=audited_rollout(self.checkpoint,goal,(d0,d1),cycles=8,delay=D)
                            self.assertEqual(audit['steps'],16)
                            self.assertEqual(audit['D'],D)
                            self.assertLessEqual(max(abs(a-b) for a,b in zip(states[-1],goal)),
                                                 3*self.cert.bound(
                                                  max(abs(d0/WEIGHTS[0]),abs(d1/WEIGHTS[1])),
                                                  self.eta,16//(2*(1+(D+1)//2))))

    def test_finite_horizon_certified_goal_and_abstention(self):
        self.assertFalse(guaranteed_absolute_goal(self.cert,self.eta,4,0))
        self.assertTrue(guaranteed_absolute_goal(self.cert,self.eta,8,0))
        self.assertFalse(guaranteed_absolute_goal(self.cert,self.eta,8,2))
        self.assertTrue(guaranteed_absolute_goal(self.cert,self.eta,16,2))
        self.assertGreater(3*self.cert.radius(self.eta),Q(1,10))
        self.assertLess(3*self.cert.radius(self.eta),Q(1,5))

    def test_reject_out_of_scope_damage(self):
        with self.assertRaisesRegex(ValueError,'damage'):
            rollout(self.W,(Q(0),Q(0)),(Q(2),Q(0)))

    def test_stability_fails_for_mismatch_even_if_empirical_samples_look_ok(self):
        with self.assertRaises(ValueError):
            Certificate.synthesize(((0,3),(Q(1,2),0)))

    def test_frozen_roster_and_test_reproducibility(self):
        a=roster(n=8)
        self.assertEqual(a,roster(n=8))
        self.assertNotEqual(a,roster(seed=3,n=8))
        r=evaluate(self.checkpoint,n=12)
        self.assertEqual(r['certificate']['trajectory_audits'],24)
        self.assertEqual(r['methods']['learned_fresh']['8']['n'],12)
        self.assertEqual(r['certificate']['certified_first_cycles_for_abs_goal'],{'0':8,'2':16})


if __name__=='__main__': unittest.main()
