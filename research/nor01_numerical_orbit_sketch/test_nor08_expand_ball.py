"""NOR-08 exact-rational second-basin radius search controls."""
import unittest
from fractions import Fraction as Q

from research.nor01_numerical_orbit_sketch import nor08_expand_ball as e


class SecondAttractorRadiusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=e.run()

    def test_frozen_prior_certificate_and_candidate_grid(self):
        ck,center,R,receipt=e.verified_source()
        self.assertEqual(ck["parameters"],1000)
        self.assertEqual(len(center),12)
        self.assertEqual(len(receipt),64)
        self.assertEqual(R,Q(1,10**20))
        self.assertEqual(tuple(sorted(e.RADII,reverse=True)),e.RADII)

    def test_largest_certified_ball_passes_every_rational_gate(self):
        out=self.result
        self.assertEqual(out["schema"],"nor08.trained_gru.second_ball_expansion.v1")
        self.assertTrue(out["certified_expanded_ball"])
        chosen=out["selected_largest_certified"]
        self.assertTrue(chosen["all_rational_gates_passed"])
        R=Q(chosen["candidate_radius_fraction"])
        K=Q(chosen["block_K_fraction"])
        s=Q(chosen["block_center_residual_fraction"])
        self.assertTrue(0<K<1)
        self.assertEqual(R-K*R-s,Q(chosen["invariance_margin_fraction"]))
        self.assertGreater(R-K*R-s,0)
        self.assertLess(Q(chosen["one_step_fixed_image_upper_fraction"]),R)
        self.assertGreater(Q(chosen["coordinate_separation_lower_fraction"]),0)
        self.assertGreater(R,Q(out["initial_nor07_radius_fraction"]))
        self.assertEqual(chosen,
            next(x for x in out["candidate_attempts"] if x["all_rational_gates_passed"]))

    def test_failed_candidates_are_explicitly_preserved(self):
        out=self.result
        self.assertEqual(len(out["candidate_attempts"]),len(e.RADII))
        for item in out["candidate_attempts"]:
            self.assertIn("candidate_radius_fraction",item)
            self.assertIn("all_rational_gates_passed",item)
        self.assertFalse(out["squared_family_source_entry_certified"])

    def test_run_is_deterministic_and_source_locked(self):
        self.assertEqual(self.result,e.run())
        self.assertEqual(self.result["checkpoint_sha256"],e.ORIGINAL_SHA)


if __name__=="__main__":
    unittest.main()
