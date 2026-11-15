"""NOR-08B: exact rational squared-source capture regression controls."""
import unittest
from fractions import Fraction as Q

from research.nor01_numerical_orbit_sketch import nor08_squared_capture as proof
from research.nor01_numerical_orbit_sketch import nor06_capture as baseline
from research.rrc_compactification import bench25


class CertifiedSquaredCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ck,digest=bench25.checkpoints()
        cls.weights=baseline.get_weights(ck["recurrent"])
        cls.ck_sha=digest["recurrent"]

    def test_independent_certified_target_radius(self):
        center,R,K,s=proof.frozen_second_basin()
        self.assertEqual(self.ck_sha,proof.SHA)
        self.assertEqual(len(center),12)
        self.assertEqual(R,Q(1,10**6))
        self.assertLess(K,1)
        self.assertGreater(R-R*K-s,0)

    def test_original_squared_source_interval(self):
        state,r,meta=baseline.source_state(self.weights,"squared")
        self.assertEqual(meta["family"],"squared")
        self.assertEqual(len(state),12)
        self.assertGreaterEqual(meta["public_header_candidates"],1)
        self.assertGreater(r,0)
        self.assertLess(r,Q(1,10**12))

    def test_first_block_is_proven_enclosure(self):
        c,r,_=baseline.source_state(self.weights,"squared")
        nxt,radius,k,e=baseline.one_block(self.weights,c,r)
        self.assertEqual(len(nxt),12)
        self.assertGreater(k,0)
        self.assertGreaterEqual(radius,e)
        self.assertGreaterEqual(radius,k*r)

    def test_short_run_marks_entry_inconclusive(self):
        out=proof.run(max_blocks=1,record_stride=1)
        self.assertEqual(out["source_metadata"]["family"],"squared")
        self.assertEqual(out["checkpoint_sha256"],self.ck_sha)
        self.assertEqual(out["certification"],"not_established")
        self.assertIsNone(out["certified_capture"])
        self.assertFalse(out["floating_point_acceptance"])
        self.assertEqual(out["block_stride"],12)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            proof.run(max_blocks=0)
        with self.assertRaises(ValueError):
            proof.run(max_blocks=129)
        with self.assertRaises(ValueError):
            proof.run(max_blocks=5,record_stride=0)


if __name__=="__main__":
    unittest.main()
