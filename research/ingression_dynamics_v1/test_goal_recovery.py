import unittest
from fractions import Fraction as Q
from research.ingression_dynamics_v1.goal_recovery import (
    goal_visible, min_goal_classes, goal_bits_lower_bound,
    goal_radius_absolute, hit_probability,
)

class GoalRecoveryTests(unittest.TestCase):
    def test_goal_visibility_and_lower_bound(self):
        states=(0,1,2,3)
        p={0:0,1:0,2:1,3:1}
        g1={0:0,1:0,2:1,3:1}
        g2={0:0,1:1,2:0,3:1}
        self.assertTrue(goal_visible(states,p,[g1]))
        self.assertFalse(goal_visible(states,p,[g1,g2]))
        self.assertEqual(min_goal_classes(states,[g1,g2]),4)
        self.assertEqual(goal_bits_lower_bound(states,[g1,g2]),2)
        self.assertEqual(goal_radius_absolute(states,p,g1),Q(0))
        self.assertEqual(goal_radius_absolute(states,p,g2),Q(1,2))

    def test_three_step_hit_probability_and_tv_bound(self):
        micro={0:{0:Q(3,4),2:Q(1,4)},
               1:{1:Q(1,2),2:Q(1,2)},
               2:{2:Q(1)}}
        macro={0:{0:Q(5,8),1:Q(3,8)},1:{1:Q(1)}}
        reference=hit_probability(macro,0,{1},3)
        self.assertEqual(reference,Q(387,512))
        for initial,expected in ((0,Q(37,64)),(1,Q(7,8))):
            actual=hit_probability(micro,initial,{2},3)
            self.assertEqual(actual,expected)
            self.assertLessEqual(abs(actual-reference),Q(169,512))

    def test_zero_horizon_and_already_hit(self):
        k={0:{1:Q(1)},1:{1:Q(1)}}
        self.assertEqual(hit_probability(k,0,{1},0),0)
        self.assertEqual(hit_probability(k,0,{1},1),1)
        self.assertEqual(hit_probability(k,1,{1},0),1)

if __name__=='__main__':
    unittest.main()
