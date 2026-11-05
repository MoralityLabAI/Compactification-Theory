import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import vicreg
from run import ARMS, main, transition_features


class TrainingTests(unittest.TestCase):
    def test_vicreg_anti_collapse(self):
        torch.manual_seed(1)
        z = torch.randn(32, 8)
        collapsed = torch.zeros_like(z)
        self.assertGreater(vicreg(collapsed, collapsed).item(), vicreg(z, z).item())

    def test_transition_features(self):
        z = torch.tensor([[1., 2.], [3., 4.]])
        f = transition_features(z, [0,1], 2)
        self.assertEqual(f.shape, (2, 8))
        self.assertEqual(f[0,-4:].tolist(), [1.,2.,0.,0.])

    def test_four_arm_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            receipts = main(['--worlds','12','--steps','5', '--epochs','1', '--batch','16',
                             '--dim','8', '--threads','1', '--seeds','11', '--out',d])
            self.assertEqual({r['arm'] for r in receipts}, set(ARMS))
            for r in receipts:
                self.assertTrue(0 <= r['metrics']['retrieval_r1'] <= 1)
                self.assertTrue(torch.isfinite(torch.tensor(r['metrics']['closure_probe_nmse'])))
            self.assertTrue((Path(d) / 'summary.json').exists())
            summary = json.loads((Path(d) / 'summary.json').read_text())
            self.assertIn('good_minus_bad_retrieval', summary['paired_deltas'])


if __name__ == '__main__':
    unittest.main()
