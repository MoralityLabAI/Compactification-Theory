import unittest
from research.shaft_compactification.capacity_finite import (
    V,STATES,immediate_signature,duplicate_signature,history_signature,reference
)

class CapacityFiniteTests(unittest.TestCase):
    def test_384_states(self):
        self.assertEqual(len(STATES),384)
    def test_exact_immediate_duplicate_equivalence(self):
        for family in ('lag_rotate','lag_match'):
            observed={}
            for state in STATES:
                observed.setdefault(immediate_signature(state,family),set()).add(duplicate_signature(state,family))
            self.assertTrue(all(len(v)==1 for v in observed.values()))
    def test_duplicate_does_not_refine_classes(self):
        r=reference()
        for f in r['families'].values():
            self.assertEqual(f['immediate_behavior_classes'],f['duplicated_immediate_classes'])
            self.assertEqual(f['immediate_behavior_classes'],64)
    def test_history_changes_classes(self):
        r=reference()
        self.assertEqual(r['families']['lag_rotate']['joint_immediate_history_classes'],348)
        self.assertEqual(r['families']['lag_match']['joint_immediate_history_classes'],384)
        self.assertTrue(all(f['immediate_alias_split_by_history_witness'] is not None for f in r['families'].values()))
    def test_two_step_teacher_forgets_previous_input(self):
        r=reference()
        self.assertTrue(all(f['two_step_history_independent_of_start_previous'] for f in r['families'].values()))

if __name__=='__main__':unittest.main()
