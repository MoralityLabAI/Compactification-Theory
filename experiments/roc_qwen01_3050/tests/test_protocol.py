import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import haar,dark_frame,route,normed,rank_p,holm,stable
from synthetic_verifier import verify


class ProtocolTests(unittest.TestCase):
    def test_refuses_wrong_gpu_before_torch_or_model_loading(self):
        from preflight import audit
        with patch("subprocess.check_output",return_value="0, GPU-wrong, NVIDIA RTX 5080, 16384, 0, 999\n"):
            with self.assertRaisesRegex(RuntimeError,"RTX 3050"):
                audit()

    def test_cpu_optimizer_fits_task_with_orthonormal_basis(self):
        from fit import optimize
        rng=np.random.default_rng(123)
        x=rng.normal(size=(32,12))
        y=x[:,0]
        u,_=optimize(x,y,haar(12,3,"init"))
        np.testing.assert_allclose(u.T@u,np.eye(3),atol=1e-10)
        prediction=(x@u)@np.linalg.lstsq(x@u,y,rcond=None)[0]
        self.assertLess(np.mean((prediction-y)**2),0.05)

    def test_exact_synthetic_scope(self):
        receipt=verify()
        self.assertEqual(receipt["cases"],9)
        self.assertFalse(receipt["neural_claims"])

    def test_donor_arms_have_declared_targets_and_equal_norm(self):
        u=haar(32,7,"observer")
        v=dark_frame(u,"dark")
        z=np.arange(32,dtype=float)
        for arm in ("dark_to_observed","observed_to_dark","observed_to_observed"):
            d=route(u,v,z,arm,0.5)
            self.assertAlmostEqual(np.linalg.norm(d),0.5,places=12)
            frame=v if arm=="observed_to_dark" else u
            np.testing.assert_allclose(frame@(frame.T@d),d,atol=1e-12)
        np.testing.assert_allclose(u.T@v,0,atol=1e-12)

    def test_haar_reproducible_and_odd_rank(self):
        a=haar(32,7,"same")
        np.testing.assert_array_equal(a,haar(32,7,"same"))
        np.testing.assert_allclose(a.T@a,np.eye(7),atol=1e-12)
        self.assertGreater(np.linalg.norm(a-haar(32,7,"other")),1)

    def test_add_one_ties_and_abstention(self):
        self.assertEqual(rank_p(200,np.arange(128)),1/129)
        self.assertEqual(rank_p(-1,np.arange(128),upper=False),1/129)
        self.assertEqual(rank_p(127,np.arange(128)),2/129)
        self.assertIsNone(rank_p(1,np.ones(128)))
        self.assertIsNone(rank_p(1,np.arange(8)))

    def test_holm_six_directions_including_unavailable(self):
        result=holm([0.001,0.01,0.03,None,None,None])
        np.testing.assert_allclose(result[:3],[0.006,0.05,0.12])
        self.assertEqual(result[3:],[None,None,None])

    def test_finite_difference_stability_and_zero_direction(self):
        self.assertTrue(stable(1,1.1))
        self.assertFalse(stable(1,2))
        self.assertTrue(stable(0,0.01))
        with self.assertRaises(ValueError):
            normed(np.zeros(4),0.5)


if __name__=="__main__":
    unittest.main()
