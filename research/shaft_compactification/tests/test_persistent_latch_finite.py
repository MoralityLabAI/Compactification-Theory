import unittest
from itertools import product
from research.shaft_compactification.persistent_latch_finite import (
    V, GOALS, STATES, step, one, two, goal_code, rollout, exact_reference
)

class PersistentLatchFiniteTests(unittest.TestCase):
    def test_state_space(self):
        self.assertEqual(len(STATES),768)
    def test_output_precedes_latch_update(self):
        state,y=step((0,0,0),0,'repeat_toggle')
        self.assertEqual(y,0)
        self.assertEqual(state,(0,1,0))
        state,y=step(state,2,'repeat_toggle')
        self.assertEqual(y,1)
    def test_same_goal_latch_one_step_alias(self):
        for g,l in product(range(GOALS),(0,1)):
            a=(g,l,0)
            for p in range(V):
                b=(g,l,p)
                for fam in ('repeat_toggle','successor_toggle'):
                    self.assertEqual(one(a,fam),one(b,fam))
                    self.assertEqual(goal_code(a),goal_code(b))
    def test_two_query_separates_all_previous_symbols(self):
        for fam in ('repeat_toggle','successor_toggle'):
            for g,l in product(range(GOALS),(0,1)):
                self.assertEqual(len({two((g,l,p),fam) for p in range(V)}),6)
    def test_shared_thirty_two_step_persistence(self):
        for fam in ('repeat_toggle','successor_toggle'):
            a,b=(13,1,0),(13,1,1)
            first=0 if fam=='repeat_toggle' else 1
            word=[first]+[2]*31
            self.assertNotEqual(one(rollout(a,word,fam),fam),one(rollout(b,word,fam),fam))
    def test_partition_counts_both_families(self):
        r=exact_reference()
        for fam,cell in r['families'].items():
            self.assertEqual([cell[k] for k in (
                'goal_signature_classes','immediate_signature_classes',
                'two_query_signature_classes',
                'joint_goal_immediate_two_query_classes')],[64,64,384,768])
            self.assertEqual(cell['pairs_two_query_distinguishes'],3840)
            self.assertEqual(cell['pairs_persist_after_32_shared_steps'],3840)

if __name__=='__main__':
    unittest.main()
