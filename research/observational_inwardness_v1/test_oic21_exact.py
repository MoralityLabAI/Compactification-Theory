"""OIC-21 finite exact adaptive query experiments, pure standard library."""
import unittest
from itertools import product
from math import gcd
from oic21_exact import ExactActiveOracle
from oic21_policy import (evaluate_policy,greedy_policy,entropy_policy,
                          fixed_marker_policy,FrozenFixedPolicy,fixed_multisensor_accuracy)
from oic20_markers import trace_distribution,partitions


class OIC21ExactTests(unittest.TestCase):
    def test_partition_conservation_and_terminal_mass(self):
        for n in (2,3,4,5,6):
            o=ExactActiveOracle(n,4)
            for t in range(4):
                for marker in range(n):
                    yes,no=o.split(o.all,t,marker)
                    self.assertEqual(yes&no,0)
                    self.assertEqual(yes|no,o.all)
                    self.assertEqual(yes.bit_count(),n)
                    self.assertEqual(no.bit_count(),n*n-n)
            self.assertEqual(o.terminal(o.all),n)

    def test_bellman_base_and_one_step(self):
        for n in (2,3,4,5):
            for q in (0,1,2):
                o=ExactActiveOracle(n,q)
                if q==0:self.assertEqual(o.adaptive_accuracy(),1/n)
                else:
                    values=[o.best_adaptive(*((o.all,0))) / (n*n)]
                    self.assertEqual(len(values),1)
                    self.assertTrue(0<values[0]<=1)

    def test_strict_adaptive_advantage_five_queries(self):
        expected={4:(15,13),5:(19,17),6:(22,18)}
        for n,(adaptive,fixed) in expected.items():
            o=ExactActiveOracle(n,5)
            self.assertEqual(o.best_adaptive(o.all,0),adaptive)
            _,actual=o.best_fixed()
            self.assertEqual(actual,fixed)
            self.assertGreater(adaptive,fixed)

    def test_exact_success_minimum_n2_to_n6(self):
        # Exhaustive dynamic programs, giving finite observations rather
        # than a general formula for arbitrary cycle lengths.
        for n in range(2,6):
            length=2*n-2
            last=ExactActiveOracle(n,length-1)
            first=ExactActiveOracle(n,length)
            self.assertLess(last.adaptive_accuracy(),1)
            self.assertEqual(first.adaptive_accuracy(),1)

    def test_n6_finite_truncated_oracle(self):
        o=ExactActiveOracle(6,8)
        self.assertEqual(o.best_adaptive(o.all,0),34)

    def test_all_zero_branch_lower_bound(self):
        # Every query forbids at most one initial phase per candidate d.
        for n in range(3,7):
            for q in range(n):
                o=ExactActiveOracle(n,q)
                for word in product(range(n),repeat=q):
                    mask=o.all
                    for t,m in enumerate(word):
                        yes,no=o.split(mask,t,m)
                        mask=no
                    for dm in o.d_masks:
                        self.assertGreaterEqual((mask&dm).bit_count(),n-q)

    def test_shift_translation_gauge(self):
        for n in range(2,8):
            for shift in range(n):
                for z in range(n):
                    for delta in range(n):
                        for t in range(2*n):
                            marker=(2*t+1)%n
                            left=int((z+shift*t)%n==marker)
                            right=int(((z+delta)+shift*t)%n==(marker+delta)%n)
                            self.assertEqual(left,right)

    def test_distribution_identification_is_not_single_episode_identification(self):
        # n=6, sensors 0 and 2 identify distinct *laws* for all shifts.
        self.assertEqual(len(partitions(6,(0,2))),6)
        self.assertEqual(fixed_multisensor_accuracy(6,(0,2),6)['accuracy'],28/36)
        # Fixed shift d=0,z=1 and d=3,z=1 both evade the two sensors.
        for d in (0,3):
            seq=tuple(tuple(int((1+d*t)%6==m) for m in (0,2)) for t in range(6))
            self.assertEqual(seq,((0,0),)*6)

    def test_adaptive_exact_population_replay(self):
        for n in (4,5,6):
            o=ExactActiveOracle(n,n+2)
            a=evaluate_policy(n,n+2,greedy_policy)
            self.assertLessEqual(a['shift_accuracy'],o.adaptive_accuracy())
            self.assertEqual(a['output_bits'],n+2)
            self.assertEqual(a['adaptive_address_bits'],(n+2)*(n-1).bit_length())
            self.assertGreaterEqual(a['conditional_entropy_bits'],0)
            self.assertLessEqual(a['brier'],1)

    def test_passive_control_is_prior(self):
        for n in (3,4,8,16):
            o=ExactActiveOracle(n,0)
            self.assertEqual(o.adaptive_accuracy(),1/n)
            self.assertEqual(o.prior_accuracy(),1/n)

    def test_input_validation(self):
        for n,q in ((1,3),(3,-1),(0,0)):
            with self.assertRaises(ValueError):ExactActiveOracle(n,q)
        self.assertEqual(evaluate_policy(3,0,greedy_policy)['shift_accuracy'],1/3)

if __name__=='__main__':unittest.main()
