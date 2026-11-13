import unittest
from research.shaft_compactification.temporal_probe_reference import (
    GOALS,QUERIES,native_output,one_step,two_step,reference
)
class TemporalProbeReferenceTests(unittest.TestCase):
    def test_alias_breaks_after_two(self):
        self.assertEqual(one_step(1,0,"xor_lag"),one_step(2,1,"xor_lag"))
        self.assertNotEqual(two_step(1,0,"xor_lag"),two_step(2,1,"xor_lag"))
    def test_previous_token_invariance(self):
        for fam in ("xor_lag","parity_gate"):
            for goal in GOALS:
                for p in QUERIES:
                    self.assertEqual(two_step(goal,0,fam),two_step(goal,p,fam))
    def test_two_step_goal_separates(self):
        for fam in ("xor_lag","parity_gate"):
            self.assertEqual(len({two_step(g,0,fam) for g in GOALS}),16)
    def test_families_differ(self):
        self.assertNotEqual(one_step(1,2,"xor_lag"),one_step(1,2,"parity_gate"))
    def test_receipt(self):
        x=reference()
        for fam in x["families"]:
            assert x["families"][fam]["physical_states"]==64
            assert x["families"][fam]["one_step_alias_pairs_split_by_two_steps"]>0
    def test_invalid(self):
        with self.assertRaises(ValueError):native_output(0,0,0,"unknown")
if __name__=="__main__":unittest.main()
