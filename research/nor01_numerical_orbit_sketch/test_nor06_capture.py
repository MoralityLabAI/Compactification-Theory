"""NOR-06B exact-recentered interval shadowing unit tests."""
import unittest
from fractions import Fraction as Q

from research.rrc_compactification import bench25,compound25
from research.nor01_numerical_orbit_sketch import nor06_basin_survey as survey
from research.nor01_numerical_orbit_sketch import nor06_capture as cap
from research.nor01_numerical_orbit_sketch.trained_gru_cert import (
    gates, get_weights, interval_forward,
)
from research.nor01_numerical_orbit_sketch.invariant_ball import CENTER,RADIUS


class VerifiedBasinCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,dig=bench25.checkpoints()
        cls.ck=ck["recurrent"]
        cls.weights=get_weights(cls.ck)
        cls.sha=dig["recurrent"]

    def test_source_and_initial_interval(self):
        c,r,meta=cap.source_state(self.weights,"signed")
        self.assertEqual(len(c),12)
        self.assertGreaterEqual(r,0)
        self.assertLess(r,Q(1,10**14))
        self.assertEqual(meta["public_header_candidates"],1)

    def test_recentered_block_covers_source_direct_interval(self):
        c,r,_=cap.source_state(self.weights,"signed")
        d,rr,K,e=cap.one_block(self.weights,c,r)
        self.assertEqual(len(d),12)
        self.assertGreater(K,0)
        self.assertGreaterEqual(rr,e)
        self.assertGreaterEqual(rr,K*r)
        cases={x.family:x for x in survey.public_cases()}
        case=cases["signed"]
        features=compound25.features(case,compound25.possible_headers(case)[0])
        direct=interval_forward(self.weights,features)
        for _ in range(cap.BLOCK):
            direct=gates(self.weights,direct)[0]
        for true_interval,cnext in zip(direct,d,strict=True):
            # Independently sound enclosures of the same true state must
            # overlap. The direct interval may be looser than the radius
            # envelope, so full inclusion would be an invalid requirement.
            self.assertLessEqual(cnext-rr,true_interval.hi)
            self.assertGreaterEqual(cnext+rr,true_interval.lo)

    def test_short_source_capture_does_not_claim_false_success(self):
        r=cap.run(family="signed",max_blocks=2,record_stride=1)
        self.assertEqual(r["checkpoint_sha256"],self.sha)
        self.assertEqual(r["actual_processed_blocks"],2)
        self.assertIsNone(r["certified_capture"])
        self.assertEqual(r["certification"],"not_established")
        self.assertTrue(r["exact_trace_sha256"])

    def test_negative_invalid_budgets(self):
        with self.assertRaises(ValueError):
            cap.run(family="signed",max_blocks=0)
        with self.assertRaises(ValueError):
            cap.run(family="signed",max_blocks=513)
        with self.assertRaises(ValueError):
            cap.unit_cube_center_ball([Q(2)]*12,Q(1,100))


if __name__=="__main__":
    unittest.main()
