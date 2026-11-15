"""NOR-08 locked second-attractor radius and squared-source controls."""
import unittest
from fractions import Fraction as Q
from dataclasses import replace

from research.nor01_numerical_orbit_sketch import nor08_radius_survey as nor08
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch import nor07_stationary_tube as tube
from research.nor01_numerical_orbit_sketch import trained_gru_cert as native


class SecondBasinRadiusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ck,cls.center=nor08.source_lock()
    def test_source_lock_and_attractor(self):
        self.assertEqual(len(self.center),12)
        self.assertEqual(nor08.SOURCE_SHA,survey.KNOWN_SHA)
        self.assertEqual(nor08.RADII[0],Q(1,10**20))
        self.assertEqual(len(nor08.RADII),10)

    def test_retain_earlier_certified_second_ball(self):
        proof=tube.stationary_tube(native.get_weights(self.ck),self.center,Q(1,10**20),24)
        self.assertTrue(proof["contracting_invariant_ball"])
        self.assertLess(Q(proof["block_operator_norm_upper_fraction"]),1)
        self.assertGreater(Q(proof["invariance_margin_fraction"]),0)

    def test_public_input_mutation_leaves_native_hidden_unchanged(self):
        candidate=next(x for x in survey.public_cases() if x.family=="squared")
        original=survey.prefix_hidden(candidate,self.ck)
        changed=replace(candidate,program=None,future=(999999,)*len(candidate.future))
        mutated=survey.prefix_hidden(changed,self.ck)
        self.assertEqual(original,mutated)

    def test_survey_short_is_discovery_only(self):
        out=nor08.run(max_steps=24,scan=False)
        self.assertEqual(out["measured"]["source_family"],"squared")
        self.assertFalse(out["source_to_second_basin_certified"])
        self.assertFalse(out["measured"]["interval_certified_entry"])
        self.assertEqual(out["max_steps"],24)
        self.assertTrue(out["two_attractors_certified"])

    def test_invalid_horizon_rejected(self):
        with self.assertRaises(ValueError):
            nor08.run(max_steps=12,scan=False)
        with self.assertRaises(ValueError):
            nor08.run(max_steps=12001,scan=False)


if __name__=="__main__":
    unittest.main()
