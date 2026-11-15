"""SMC-12 exact cone self-selection, source tamper and ray provenance tests."""
from fractions import Fraction as Q
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from conic_self_selection_exact import (
    physical_encoder_normals,cone_certificate,relative_raw_guard,
    unit_chord_guard,joint_normal_and_decoder_guard,
    sharp_boundary_adversary
)
from smc07_real_sae import TopKSAE
from smc11_selection import frozen_active_direction_dyadics
from smc12_cones import load_frozen_sources,sae_hash,EXPECTED_FITS


class FrozenConicTests(unittest.TestCase):
    def test_exact_simple_rational_margin_and_adversarial_boundary(self):
        normal=((1,0),(0,1),(-1,0))
        c=cone_certificate(normal,(0,),(2,1))
        self.assertEqual(c['status'],'interior')
        self.assertEqual(c['minimal_normalized_slope_squared'],Q(1,10))
        self.assertTrue(relative_raw_guard(c,Q(3,10)))
        self.assertFalse(relative_raw_guard(c,Q(1,3)))
        self.assertTrue(unit_chord_guard(c,Q(3,10)))
        self.assertFalse(unit_chord_guard(c,Q(1,3)))
        self.assertTrue(joint_normal_and_decoder_guard(c,Q(1,5),Q(1,20)))
        w=sharp_boundary_adversary(normal,(0,),(2,1),c)
        self.assertEqual(w['exact_boundary_ray'],(Q(3,2),Q(3,2)))
        self.assertLess(w['outside_gap_slope'],0)

    def test_physical_normalization_matters(self):
        rows=((2,0),(0,6),(-3,0))
        a=physical_encoder_normals(rows,(2,3))
        b=physical_encoder_normals(rows,(1,1))
        self.assertNotEqual(a,b)
        self.assertNotEqual(
            cone_certificate(a,(0,),(2,1))['minimal_normalized_slope_squared'],
            cone_certificate(b,(0,),(2,1))['minimal_normalized_slope_squared'])

    def test_frozen_tensor_hash_and_direction_convention(self):
        torch.manual_seed(77)
        model=TopKSAE(48,96,4).eval()
        scale=torch.rand(48,dtype=torch.float32)+0.3
        mu=torch.rand(48,dtype=torch.float32)
        h=sae_hash(model,mu,scale)
        self.assertEqual(len(h),64)
        self.assertEqual(sae_hash(model,mu,scale),h)
        direction=frozen_active_direction_dyadics(model,scale,3)
        self.assertEqual(len(direction),48)
        reference=(model.decoder.weight[:,3]*scale).detach().double()
        reference=reference/reference.norm()
        self.assertEqual(direction,tuple(reference.cpu().tolist()))
        with torch.no_grad():
            model.encoder.bias[0]+=0.05
        self.assertNotEqual(sae_hash(model,mu,scale),h)

    def test_untrusted_artifact_rejected_before_training_snapshot_load(self):
        with tempfile.TemporaryDirectory() as root:
            f=Path(root)/'falsified.zip'
            f.write_bytes(b'PK\x03\x04bad bytes')
            with self.assertRaisesRegex(ValueError,'ZIP SHA256'):
                load_frozen_sources(archive=f)
            with self.assertRaises((FileNotFoundError,ValueError)):
                load_frozen_sources(extracted_dir=Path(root))
            with self.assertRaises(ValueError):
                load_frozen_sources()

    def test_all_frozen_sha256_pins_distinct(self):
        self.assertEqual(set(EXPECTED_FITS),{71,89,113})
        hashes=[x for pair in EXPECTED_FITS.values() for x in pair]
        self.assertEqual(len(hashes),len(set(hashes)))
        self.assertTrue(all(len(s)==64 for s in hashes))


if __name__=="__main__":
    unittest.main()
