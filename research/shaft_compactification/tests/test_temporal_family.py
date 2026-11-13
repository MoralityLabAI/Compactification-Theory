import unittest
from research.shaft_compactification.temporal_family_reference import (
    current_reply, family_trace, reference
)

class TemporalFamilyTests(unittest.TestCase):
    def test_rotate_uses_previous_token(self):
        self.assertEqual(current_reply(1,0,0,'lag_rotate'),1)
        self.assertEqual(current_reply(1,0,1,'lag_rotate'),0)

    def test_match_uses_previous_token(self):
        self.assertEqual(current_reply(0,0,0,'lag_match'),1)
        self.assertEqual(current_reply(0,0,1,'lag_match'),0)

    def test_first_previous_sentinel_zero(self):
        for goal in range(16):
            for token in range(4):
                for family in ('lag_rotate','lag_match'):
                    self.assertEqual(family_trace(goal,(token,),family),
                                     (current_reply(goal,token,0,family),))

    def test_output_is_binary_for_all_short_histories(self):
        for family in ('lag_rotate','lag_match'):
            for goal in range(16):
                for a in range(4):
                    for b in range(4):
                        self.assertIn(family_trace(goal,(a,b),family)[1],(0,1))

    def test_unobserved_previous_changes_both_task_families(self):
        record=reference()
        for family in ('lag_rotate','lag_match'):
            self.assertGreater(record['family_sensitivity'][family]
                               ['goal_current_combinations_history_sensitive'],0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):family_trace(16,(1,),'lag_rotate')
        with self.assertRaises(ValueError):family_trace(0,(4,),'lag_match')
        with self.assertRaises(ValueError):family_trace(0,(1,),'unknown')
        with self.assertRaises(ValueError):current_reply(0,1,4,'lag_rotate')

if __name__=='__main__':
    unittest.main()
