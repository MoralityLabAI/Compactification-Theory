"""NOR-06 model-locked, nonleaky finite basin-entry survey tests."""
import unittest
from dataclasses import replace

from research.rrc_compactification import bench25,compound25
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as s


class NOR06PublicBasin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,dig=bench25.checkpoints()
        cls.model=ck["recurrent"]
        cls.sha=dig["recurrent"]
        cls.cases=s.public_cases()

    def test_frozen_source_checkpoint_and_cohort(self):
        self.assertEqual(self.sha,s.KNOWN_SHA)
        self.assertEqual(len(self.cases),6)
        self.assertEqual(tuple(x.family for x in self.cases),s.FAMILIES)
        self.assertEqual(len(set(x.visible for x in self.cases)),6)

    def test_hidden_state_uses_public_input_only(self):
        case=self.cases[0]
        initial=s.prefix_hidden(case,self.model)
        mutant=replace(case,identifier="id-does-not-matter",program=None,
                       future=tuple(-1000000 for _ in case.future))
        other=s.prefix_hidden(mutant,self.model)
        self.assertEqual(initial,other)

    def test_exact_native_head_parity(self):
        for case in self.cases:
            h,parity,nh=s.prefix_hidden(case,self.model)
            self.assertEqual(len(h),12)
            self.assertLess(parity,1e-11)
            self.assertGreaterEqual(nh,1)

    def test_zero_continuation_deterministic_and_explicitly_uncertified(self):
        a=s.run(max_steps=24)
        b=s.run(max_steps=24)
        self.assertEqual(a,b)
        self.assertFalse(a["limits"]["certified_entry"])
        self.assertFalse(a["limits"]["input_truth_or_future_used"])
        self.assertEqual(a["family_count"],6)
        self.assertTrue(all(x["initial_center_distance"]>0 for x in a["rows"]))
        self.assertTrue(all(x["source_native_logits_parity"]<1e-11 for x in a["rows"]))

    def test_reject_unbounded_sampling_budget(self):
        with self.assertRaises(ValueError):
            s.run(max_steps=12)
        with self.assertRaises(ValueError):
            s.run(max_steps=12001)


if __name__=="__main__":
    unittest.main()
