"""SMC-13B: independent optimal assignment and puzzle-cluster holdout tests."""
import itertools
import math
import random
import sys
import unittest
from pathlib import Path

import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smc13_transfer import (
    hungarian_maximum,center_affinity,heldout_overlap,
    puzzle_bootstrap,null_permutations,evaluate_alignment
)


class CrossTeacherTransferTests(unittest.TestCase):
    def test_hungarian_exactly_matches_brute_force_optimum(self):
        rng=random.Random(202613)
        for n in range(2,7):
            for _ in range(12):
                scores=[[rng.randint(-5,15) for j in range(n)]
                        for i in range(n)]
                obtained=hungarian_maximum(scores)
                truth=max(sum(scores[i][p[i]] for i in range(n))
                          for p in itertools.permutations(range(n)))
                self.assertEqual(sum(scores[i][obtained[i]] for i in range(n)),truth)
                self.assertEqual(sorted(obtained),list(range(n)))

    def test_centered_activation_covariance_removes_constant_features(self):
        a=torch.tensor([[[1.,0.,0.]],[[1.,0.,0.]]]).repeat(1,81,32)
        b=torch.tensor([[[1.,0.,0.]],[[0.,1.,0.]]]).repeat(1,81,32)
        # Construct separate nonempty 96-channel tensors with a constant
        # active source channel, whose centered covariance must be zero.
        a=torch.zeros((4,81,96));b=torch.zeros_like(a)
        a[:,:,1]=1
        b[:2,:,2]=1
        b[2:,:,3]=1
        result=center_affinity(a,b)
        self.assertEqual(len(result),96)
        self.assertAlmostEqual(result[1][2],0.0,places=10)
        self.assertAlmostEqual(result[1][3],0.0,places=10)

    def test_known_feature_permutation_recovered_out_of_sample(self):
        rng=random.Random(471)
        torch.manual_seed(471)
        a=(torch.rand((12,81,96))<0.1).float()
        order=list(range(96));rng.shuffle(order)
        # B[:, :, order[i]] = A[:, :, i]
        b=torch.zeros_like(a)
        for i,j in enumerate(order):b[:,:,j]=a[:,:,i]
        learned=hungarian_maximum(center_affinity(a[:8],b[:8]))
        self.assertEqual(learned,order)
        per_puzzle=heldout_overlap(a[8:],b[8:],learned)
        self.assertTrue(all(abs(x-1.0)<1e-9 for x in per_puzzle))
        null=null_permutations(a[8:],b[8:],draws=20)
        self.assertLess(null['mean'],0.25)

    def test_cluster_bootstrap_uses_puzzle_unit(self):
        r=puzzle_bootstrap([0.0,0.0,1.0,1.0],draws=200,seed=18)
        self.assertEqual(r['n_puzzle_clusters'],4)
        self.assertAlmostEqual(r['mean'],0.5)
        self.assertEqual(len(r['puzzle_bootstrap95']),2)

    def test_wrong_dimensions_or_alignment_fail_closed(self):
        with self.assertRaises(ValueError):hungarian_maximum([[1.,2.]])
        with self.assertRaises(ValueError):center_affinity(torch.zeros(2,3,2),torch.zeros(2,3,3))
        with self.assertRaises(ValueError):
            heldout_overlap(torch.zeros(1,81,96),torch.zeros(1,81,96),list(range(95)))
        with self.assertRaises(ValueError):puzzle_bootstrap([],draws=10)


if __name__=='__main__':
    unittest.main()
