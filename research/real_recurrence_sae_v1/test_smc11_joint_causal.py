"""SMC-11C self-contained frozen-SAE provenance and causal-guard smoke tests."""
import tempfile
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc07_real_sae import TopKSAE
from smc11_joint_causal import (
    frozen_sae_fingerprint,serialize_snapshot,same_fit_policy
)


class SameFitSAETests(unittest.TestCase):
    def test_exact_fp32_training_snapshot_roundtrip_and_fingerprint(self):
        torch.manual_seed(113)
        sae=TopKSAE(d_input=48,width=96,k=4)
        mu=torch.randn(48,dtype=torch.float32)
        scale=torch.ones(48,dtype=torch.float32)*0.7
        a=frozen_sae_fingerprint(sae,mu,scale)
        self.assertEqual(len(a),64)
        self.assertEqual(frozen_sae_fingerprint(sae,mu,scale),a)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'fit_seed113.pt'
            file_sha=serialize_snapshot(sae,mu,scale,'source_sha',113,path)
            self.assertEqual(len(file_sha),64)
            self.assertTrue(path.is_file())
            frozen=torch.load(path,weights_only=True,map_location='cpu')
            other=TopKSAE(d_input=48,width=96,k=4)
            other.load_state_dict(frozen['state_dict'])
            self.assertEqual(frozen_sae_fingerprint(other,frozen['train_mean'],
                                                      frozen['train_scale']),a)
        with torch.no_grad():
            sae.encoder.weight[0,0]+=torch.tensor(1e-3)
        self.assertNotEqual(frozen_sae_fingerprint(sae,mu,scale),a)
        self.assertNotEqual(frozen_sae_fingerprint(other,mu,scale+0.001),a)

    def test_exact_same_fit_guards_and_adaptive_reference_boundary(self):
        identity={'status':'certified',
                  'budget_guards':{
                     '0':{'top1':True,'top4':True},
                     '1/1000000':{'top1':True,'top4':False}}}
        self.assertEqual(same_fit_policy('sae_active_greater',identity,
                                        'isotropic','1/1000000'),
                         'sae_active_greater')
        self.assertEqual(same_fit_policy('control_greater',identity,
                                        'jacobian_optimal','1/1000000'),
                         'control_greater')
        self.assertEqual(same_fit_policy('sae_active_greater',identity,
                                        'sae_inactive','1/1000000'),
                         'abstain_selection')
        identity['status']='abstain_ValueError'
        self.assertEqual(same_fit_policy('sae_active_greater',identity,
                                        'isotropic','0'),'abstain_selection')
        with self.assertRaises(ValueError):
            same_fit_policy('sae_active_greater',identity,'future_control','0')


if __name__=='__main__':
    unittest.main()
