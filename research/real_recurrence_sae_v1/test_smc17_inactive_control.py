"""CPU regression tests for label-blind SMC-17 sham geometry."""
import unittest
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import smc17_inactive_control as s
import smc16_selective_kill as prior
from test_smc15_active_necessity import FakeSAE


class InactiveControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def toy(self):
        sae=FakeSAE().eval()
        with torch.no_grad():
            # Inactive fifth encoder row sees both kill e0 and tangent e4.
            # It remains inactive under the clean and modified toy states.
            sae.encoder.weight[4,0]=1
            sae.encoder.weight[4,4]=1
            sae.encoder.bias[4]=-10
        h=torch.zeros(48)
        h[:4]=torch.tensor([3.,2.,1.,0.5])
        mean=torch.zeros(48);scale=torch.ones(48)
        code=prior.encode_one(sae,mean,scale,h)
        cov=prior.physical_covectors(sae,scale)
        dec=prior.physical_columns(sae,scale)
        kill,baseline,info=prior.selective_vectors(
            h,code,0,sae,mean,scale,cov,dec,1,0,'toy',0)
        self.assertEqual(info["status"],"eligible")
        return h,sae,mean,scale,code,cov,kill,baseline

    def test_spectral_least_squares_sham_does_better_than_random(self):
        h,sae,mean,scale,z,W,kill,baseline=self.toy()
        selected,d=s.best_inactive_matched_sham(
            h,z,0,sae,mean,scale,W,kill,baseline,0,'toy',1,0)
        self.assertEqual(len(d["candidate_receipts"]),131)
        self.assertEqual(d["tangent_dimension"],44)
        self.assertEqual(len(d["inactive_ids"]),92)
        self.assertEqual(len(d["all96_best_preactivation_delta"]),96)
        self.assertGreater(d["epsilon_baseline"],0)
        self.assertLess(d["epsilon_best"],d["epsilon_baseline"])
        self.assertAlmostEqual(d["inactive_residual_floor"],0,places=8)
        self.assertAlmostEqual(d["epsilon_best"],0,places=5)
        self.assertTrue(bool(torch.allclose(kill.norm(),selected.norm(),rtol=0,atol=1e-9)))
        self.assertTrue(all(abs(d["all96_best_preactivation_delta"][j])<1e-8
                            for j in range(4)))

    def test_fixed_seeds_and_best_candidate_repeat(self):
        a,b,_,_,_,_,_,_=self.toy()
        cov=prior.physical_covectors(b,torch.ones(48))
        active=cov[[0,1,2,3]]
        x,seed1=s.random_tangent(2,"new",1000,3,7,5,active)
        y,seed2=s.random_tangent(2,"new",1000,3,7,5,active)
        self.assertEqual(seed1,seed2)
        self.assertTrue(torch.equal(x,y))
        self.assertLess(float((active@x).abs().max()),1e-8)

    def test_changed_random_budget_fails_closed(self):
        h,sae,mean,scale,z,W,kill,baseline=self.toy()
        with self.assertRaises(ValueError):
            s.best_inactive_matched_sham(
                h,z,0,sae,mean,scale,W,kill,baseline,0,"toy",1,0,random_count=64)

    def test_wrong_shapes_rejected(self):
        h,sae,mean,scale,z,W,kill,baseline=self.toy()
        with self.assertRaises(ValueError):
            s.best_inactive_matched_sham(
                h,z,0,sae,mean,scale,W[:50],kill,baseline,0,"toy",1,0)

    def test_registered_roster_counts_only(self):
        with self.assertRaises(ValueError):
            s.fresh_rosters({},limit=5)


if __name__=="__main__":
    unittest.main()
